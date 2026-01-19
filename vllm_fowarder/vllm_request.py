import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp
import pandas as pd
from tqdm import tqdm

# =========================
# 配置区
# =========================
VLLM_HOST = "::1"
VLLM_PORT = 8000
MODEL_NAME = "default"

# vLLM 的 OpenAI 兼容接口（常见路径）
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# 固定并发
CONCURRENCY = 1024

# 请求参数（按需改）
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 1.0

# 超时（秒）
TOTAL_TIMEOUT_S = 1800

# 重试策略
RETRIES = 10
BASE_BACKOFF_S = 2
MAX_BACKOFF_S = 32


def build_openai_messages(prompt_messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    你的 parquet 里 df["prompt"] 看起来是 list[dict]，每个 dict 有 role/content。
    这里直接转换成 OpenAI/vLLM 兼容的 messages: [{"role": "...", "content": "..."}]
    """
    out: List[Dict[str, str]] = []
    for item in prompt_messages or []:
        role = (item or {}).get("role", "user")
        content = (item or {}).get("content", "")
        # 兜底：vLLM 通常支持 system/user/assistant
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": content})
    return out


class PauseController:
    """
    全局退避：一旦某次请求触发退避，可让所有请求在一段时间内共同等待，
    避免雪崩式打爆服务。
    """
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pause_until = 0.0

    async def wait(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                delay = self._pause_until - now
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def pause(self, seconds: float) -> None:
        async with self._lock:
            until = time.monotonic() + seconds
            if until > self._pause_until:
                self._pause_until = until


async def post_chat_completion(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    pause_ctl: Optional[PauseController],
) -> Dict[str, Any]:
    """
    单次 HTTP POST（不含重试）
    """
    if pause_ctl is not None:
        await pause_ctl.wait()

    async with session.post(url, json=payload) as resp:
        # vLLM 出错时可能返回非 200，读出文本方便定位
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:2000]}")
        return await resp.json()


async def call_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    sem: asyncio.Semaphore,
    pause_ctl: Optional[PauseController],
) -> Dict[str, Any]:
    last_exc: Optional[BaseException] = None

    # 先等全局 pause，再拿并发锁
    if pause_ctl is not None:
        await pause_ctl.wait()

    async with sem:
        for attempt in range(1, RETRIES + 2):  # 总尝试次数 = RETRIES + 1
            try:
                if pause_ctl is not None:
                    await pause_ctl.wait()
                return await post_chat_completion(session, url, payload, pause_ctl)
            except Exception as e:
                last_exc = e
                if attempt <= RETRIES:
                    backoff = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** (attempt - 1)))
                    # 抖动，避免同一时刻齐刷刷重试
                    backoff = backoff * (0.8 + 0.4 * random.random())
                    print(f"[retry] attempt={attempt}/{RETRIES} error={repr(e)} backoff_s={backoff:.2f}")
                    if pause_ctl is not None:
                        await pause_ctl.pause(backoff)
                    await asyncio.sleep(backoff)
                else:
                    print(f"[error] attempts_exhausted={RETRIES + 1} error={repr(e)}")
                    raise last_exc


def extract_text(resp_json: Dict[str, Any]) -> str:
    """
    OpenAI/vLLM chat.completions 常见返回结构：
    {"choices":[{"message":{"role":"assistant","content":"..."}, ...}], ...}
    """
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        # 兜底：把原始响应 dump 出来，便于排查
        return json.dumps(resp_json, ensure_ascii=False)


async def run_requests(
    batch_messages: Sequence[List[Dict[str, str]]],
) -> Tuple[List[Optional[Dict[str, Any]]], int, int]:
    sem = asyncio.Semaphore(CONCURRENCY)
    pause_ctl = PauseController()

    base_url = f"http://[{VLLM_HOST}]:{VLLM_PORT}{CHAT_COMPLETIONS_PATH}"

    timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT_S)
    # 并发很大：用 TCPConnector 控制连接池上限
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)

    results: List[Optional[Dict[str, Any]]] = [None] * len(batch_messages)
    ok = 0
    fail = 0

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def one(i: int, msgs: List[Dict[str, str]]):
            payload = {
                "model": MODEL_NAME,
                "messages": msgs,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "top_p": DEFAULT_TOP_P,
                "temperature": DEFAULT_TEMPERATURE,
                "stream": False,
            }
            resp_json = await call_with_retry(
                session=session,
                url=base_url,
                payload=payload,
                sem=sem,
                pause_ctl=pause_ctl,
            )
            return i, resp_json

        tasks = [asyncio.create_task(one(i, batch_messages[i])) for i in range(len(batch_messages))]

        with tqdm(total=len(tasks), desc="sending", unit="req") as pbar:
            for fut in asyncio.as_completed(tasks):
                try:
                    i, resp_json = await fut
                    results[i] = resp_json
                    ok += 1
                except Exception:
                    fail += 1
                finally:
                    pbar.set_postfix(ok=ok, fail=fail)
                    pbar.update(1)

    return results, ok, fail


async def main():
    path = "RubricHub_v1/RubricHub_v1/RuRL/rurbichub_v1_Medical.parquet"

    df = pd.read_parquet(path)

    # df["prompt"] 每行应是 list[dict]，直接转成 OpenAI messages
    batch_messages: List[List[Dict[str, str]]] = [
        build_openai_messages(prompt) for prompt in df["prompt"].tolist()
    ]

    results, ok, fail = await run_requests(batch_messages)

    print("=" * 80)
    print(f"done. ok={ok}, fail={fail}")
    print("=" * 80)

    # 示例打印第一条
    if results and results[0] is not None:
        print("first raw response json keys:", list(results[0].keys()))
        print("=" * 80)
        print(extract_text(results[0]))
        print("=" * 80)

    # # 可选：把模型输出落盘
    # df["vllm_response_json"] = results
    # df["vllm_text"] = [extract_text(r) if r is not None else None for r in results]
    # out_path = path.replace(".parquet", ".output.parquet")
    # df.to_parquet(out_path, index=False)
    # print(f"saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
