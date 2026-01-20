import asyncio
import os
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse


METRICS_RUNNING_KEYS = [
    r"^vllm[:_]num_requests_running(?:\{|\s)",
    r"^num_requests_running(?:\{|\s)",
]
METRICS_WAITING_KEYS = [
    r"^vllm[:_]num_requests_waiting(?:\{|\s)",
    r"^num_requests_waiting(?:\{|\s)",
]


def _parse_prometheus_value(lines: List[str], patterns: List[str]) -> Optional[int]:
    for line in lines:
        for pat in patterns:
            if re.search(pat, line):
                parts = line.strip().split()
                if parts:
                    try:
                        # last token is value in typical Prometheus exposition
                        val = int(float(parts[-1]))
                        return val
                    except Exception:
                        continue
    return None


@dataclass
class Backend:
    name: str
    base_v1: str  # e.g. http://host:8101/v1
    base_root: str  # e.g. http://host:8101
    healthy: bool = False
    running: int = 0
    waiting: int = 0
    last_status: Optional[int] = None
    last_error: Optional[str] = None


class BackendRegistry:
    def __init__(self, backends: List[Backend]):
        self.backends: Dict[str, Backend] = {b.name: b for b in backends}
        self.virtual_loads: Dict[str, int] = {b.name: 0 for b in backends}
        self._lock = asyncio.Lock()

    async def list_healthy(self) -> List[Backend]:
        async with self._lock:
            return [Backend(**vars(b)) for b in self.backends.values() if b.healthy]

    async def get_all(self) -> List[Backend]:
        async with self._lock:
            return [Backend(**vars(b)) for b in self.backends.values()]

    async def update(self, name: str, **kwargs):
        async with self._lock:
            b = self.backends.get(name)
            if not b:
                return
            for k, v in kwargs.items():
                setattr(b, k, v)

    async def reconcile(self, new_backends: List[Backend]):
        """Reconcile registry with a new backend list.
        - Add new items
        - Update base URLs for existing names
        - Remove items no longer present
        """
        async with self._lock:
            current = self.backends
            new_map: Dict[str, Backend] = {b.name: b for b in new_backends}

            # Remove missing
            for name in list(current.keys()):
                if name not in new_map:
                    current.pop(name, None)
                    self.virtual_loads.pop(name, None)

            # Add or update
            for name, nb in new_map.items():
                if name in current:
                    cb = current[name]
                    cb.base_root = nb.base_root
                    cb.base_v1 = nb.base_v1
                else:
                    current[name] = nb
                    if name not in self.virtual_loads:
                        self.virtual_loads[name] = 0

    async def reserve(self, name: str):
        async with self._lock:
            self.virtual_loads[name] = self.virtual_loads.get(name, 0) + 1

    async def release(self, name: str):
        async with self._lock:
            self.virtual_loads[name] = max(0, self.virtual_loads.get(name, 0) - 1)

    async def get_virtuals(self) -> Dict[str, int]:
        async with self._lock:
            return dict(self.virtual_loads)

    async def choose_and_reserve(self, exclude: Optional[Set[str]] = None) -> Optional[Backend]:
        """Atomically choose the least-loaded healthy backend and reserve virtual load.
        Break ties randomly to avoid hot-spotting when many are equal.
        Returns a copy of the chosen backend or None if none available.
        """
        async with self._lock:
            healthy = [b for b in self.backends.values() if b.healthy and (not exclude or b.name not in exclude)]
            if not healthy:
                return None

            def key(b: Backend):
                base = (b.running or 0) + (b.waiting or 0)
                virt = self.virtual_loads.get(b.name, 0)
                return (base + virt, b.waiting or 0, b.running or 0)

            # Find minimal key and deterministically choose among ties
            min_key = None
            ties: List[Backend] = []
            for b in healthy:
                k = key(b)
                if min_key is None or k < min_key:
                    min_key = k
                    ties = [b]
                elif k == min_key:
                    ties.append(b)
            # Choose the first tie deterministically
            chosen = ties[0]
            self.virtual_loads[chosen.name] = self.virtual_loads.get(chosen.name, 0) + 1
            return Backend(**vars(chosen))


def normalize_base(line: str, default_port: int = 8101) -> Tuple[str, str, str]:
    raw = line.strip()
    if not raw:
        raise ValueError("empty backend line")
    if raw.startswith("http://") or raw.startswith("https://"):
        base = raw.rstrip("/")
    else:
        # Parse bare host[:port][/<path>], default to http
        if "/" in raw:
            hostport, rest = raw.split("/", 1)
            path = "/" + rest
        else:
            hostport, path = raw, ""

        # Detect if a port is explicitly present
        has_port = False
        if hostport.startswith("["):  # IPv6 with brackets
            m = re.match(r"^\[.*\](?::(\d+))?$", hostport)
            has_port = bool(m and m.group(1))
        else:
            idx = hostport.rfind(":")
            has_port = idx != -1 and hostport[idx + 1 :].isdigit()

        if has_port:
            base = f"http://{hostport}{path}"
        else:
            base = f"http://{hostport}:{default_port}{path}"

        base = base.rstrip("/")

    if base.endswith("/v1"):
        base_v1 = base
        base_root = base[:-3]
    else:
        base_root = base
        base_v1 = base_root + "/v1"

    name = base_root.replace("http://", "").replace("https://", "")
    return name, base_v1, base_root


def load_backends_from_file(path: str, default_port: int = 8101) -> List[Backend]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"IP list file not found: {path}")
    backends: List[Backend] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, base_v1, base_root = normalize_base(line, default_port=default_port)
            backends.append(Backend(name=name, base_v1=base_v1, base_root=base_root))
    if not backends:
        raise RuntimeError("No backends found in ip.txt")
    return backends


async def fetch_metrics(client: httpx.AsyncClient, backend: Backend, timeout: float = 3.0) -> Tuple[bool, Optional[int], Optional[int], Optional[int], Optional[str]]:
    url = backend.base_root + "/metrics"
    try:
        r = await client.get(url, timeout=timeout)
        status = r.status_code
        if status != 200:
            return False, None, None, status, f"status {status}"
        text = r.text
        lines = text.splitlines()
        running = _parse_prometheus_value(lines, METRICS_RUNNING_KEYS)
        waiting = _parse_prometheus_value(lines, METRICS_WAITING_KEYS)
        return True, running if running is not None else 0, waiting if waiting is not None else 0, status, None
    except Exception as e:
        return False, None, None, None, str(e)


def create_app() -> FastAPI:
    ip_file = os.getenv("IP_LIST_FILE", "ip.txt")
    check_interval = float(os.getenv("CHECK_INTERVAL_SECONDS", "10"))
    default_port = int(os.getenv("BACKEND_DEFAULT_PORT", "8101"))
    upstream_timeout_seconds = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "3600"))
    retry_delay_seconds = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger("vllm_forwarder")
    if not logger.handlers:
        # Basic config if not already configured by uvicorn
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Suppress noisy httpx/httpcore request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    backends = load_backends_from_file(ip_file, default_port=default_port)
    registry = BackendRegistry(backends)
    app = FastAPI(title="vLLM Forwarder", version="0.1.0")

    @app.on_event("startup")
    async def startup_event():
        async def loop():
            async with httpx.AsyncClient() as client:
                while True:
                    logger.info("%s", "=" * 60)
                    # Reload ip list each cycle (every check_interval seconds)
                    try:
                        # Track changes: added/removed
                        prev_list = await registry.get_all()
                        prev_names = {b.name for b in prev_list}
                        new_list = load_backends_from_file(ip_file, default_port=default_port)
                        new_names = {b.name for b in new_list}
                        added = sorted(new_names - prev_names)
                        removed = sorted(prev_names - new_names)
                        if added or removed:
                            logger.info(
                                "IP list updated: +%d, -%d | added=%s removed=%s",
                                len(added), len(removed), added, removed,
                            )
                        await registry.reconcile(new_list)
                    except Exception as e:
                        # Keep previous list if reading/parsing fails
                        logger.warning("Failed to reload ip list: %s", e)
                    for b in await registry.get_all():
                        ok, running, waiting, status, err = await fetch_metrics(client, b)
                        await registry.update(
                            b.name,
                            healthy=ok,
                            running=running or 0,
                            waiting=waiting or 0,
                            last_status=status,
                            last_error=err,
                        )
                    # Metrics summary logging
                    all_bs = await registry.get_all()
                    healthy_bs = [x for x in all_bs if x.healthy]
                    logger.info(
                        "Metrics summary: healthy %d/%d",
                        len(healthy_bs), len(all_bs),
                    )
                    if healthy_bs:
                        for b in healthy_bs:
                            chat = f"{b.base_v1}/chat/completions"
                            logger.info("%s (run=%d, wait=%d)", chat, b.running, b.waiting)
                    await asyncio.sleep(check_interval)

        asyncio.create_task(loop())

    @app.get("/_backends")
    async def backends_status():
        items = [
            {
                "name": b.name,
                "base_root": b.base_root,
                "base_v1": b.base_v1,
                "healthy": b.healthy,
                "running": b.running,
                "waiting": b.waiting,
                "last_status": b.last_status,
                "last_error": b.last_error,
            }
            for b in await registry.get_all()
        ]
        return {"backends": items}

    async def choose_backend(exclude: Optional[Set[str]] = None) -> Optional[Backend]:
        # Legacy chooser (non-atomic). Kept for reference/debug endpoints if needed.
        healthy = await registry.list_healthy()
        if not healthy:
            return None
        candidates = [b for b in healthy if not exclude or b.name not in exclude]
        if not candidates:
            return None
        vmap = await registry.get_virtuals()
        def load_key(b: Backend):
            base = (b.running or 0) + (b.waiting or 0)
            virt = vmap.get(b.name, 0)
            total = base + virt
            # order by total, then wait, then run; break ties randomly later
            return (total, b.waiting or 0, b.running or 0)
        # Deterministic tie handling (no randomness)
        values = [(b, load_key(b)) for b in candidates]
        if not values:
            return None
        min_key = min(v for _, v in values)
        ties = [b for b, v in values if v == min_key]
        return ties[0]

    def retriable_status(code: int) -> bool:
        return code == 429 or 500 <= code < 600

    async def proxy_request(request: Request, subpath: str) -> Response:
        # Build proxied headers (remove host, keep auth etc.)
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Body handling: we read once and reuse across retries
        body = await request.body()

        # Retry-forever loop: waits for healthy backend and retriable failures
        attempted: Set[str] = set()
        while True:
            exclude = attempted if attempted else None
            backend = await registry.choose_and_reserve(exclude=exclude)
            if not backend:
                attempted.clear()
                await asyncio.sleep(retry_delay_seconds)
                continue

            upstream_url = f"{backend.base_v1}/{subpath}" if subpath else backend.base_v1

            attempted.add(backend.name)

            timeout = httpx.Timeout(
                timeout=upstream_timeout_seconds,
                connect=min(10.0, upstream_timeout_seconds),
                read=upstream_timeout_seconds,
                write=upstream_timeout_seconds,
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    # Use streaming upstream, but decide response mode by content-type
                    req = client.build_request(
                        request.method,
                        upstream_url,
                        params=request.query_params,
                        headers=headers,
                        content=body if request.method.upper() in {"POST", "PUT", "PATCH"} else None,
                    )
                    resp = await client.send(req, stream=True)
                except Exception:
                    # Transport error; retry another backend
                    await registry.release(backend.name)
                    await asyncio.sleep(retry_delay_seconds)
                    continue

                # Decide streaming based on content-type and status
                ct = resp.headers.get("content-type", "")
                is_streaming = ("text/event-stream" in ct) or ("application/x-ndjson" in ct)

                # Retry on 429/5xx
                if retriable_status(resp.status_code):
                    await resp.aclose()
                    await registry.release(backend.name)
                    await asyncio.sleep(retry_delay_seconds)
                    continue

                if is_streaming:
                    bname = backend.name
                    async def aiter():
                        try:
                            async for chunk in resp.aiter_bytes():
                                if chunk:
                                    yield chunk
                        finally:
                            await resp.aclose()
                            await registry.release(bname)

                    passthrough_headers = {k: v for k, v in resp.headers.items() if k.lower() not in {"content-encoding", "transfer-encoding"}}
                    return StreamingResponse(aiter(), status_code=resp.status_code, headers=passthrough_headers)

                # Non-streaming: read full content and return
                content = await resp.aread()
                await resp.aclose()
                await registry.release(backend.name)
                passthrough_headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-encoding"}
                return Response(content=content, status_code=resp.status_code, headers=passthrough_headers)

    # Catch-all proxy for OpenAI-compatible API under /v1/*
    @app.api_route("/v1/{subpath:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def v1_proxy(subpath: str, request: Request):
        return await proxy_request(request, subpath)

    # Simple root and health endpoints
    @app.get("/")
    async def root():
        return PlainTextResponse("vLLM forwarder up. See /_backends for status.")

    return app


app = create_app()
