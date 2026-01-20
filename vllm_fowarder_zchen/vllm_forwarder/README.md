vLLM Forwarder (OpenAI-compatible)

A lightweight FastAPI service that forwards OpenAI-compatible `/v1/*` requests to multiple vLLM backends. It determines backend availability by scraping each backend's Prometheus metrics endpoint every 10 seconds and randomly chooses a healthy backend per request.

- Health check: GET `{base_root}/metrics` is 200 => healthy. Parses `num_requests_running` and `num_requests_waiting` if present.
- Proxy: Random healthy backend per request.
- API: Transparent proxy for `/v1/*` (chat/completions, completions, embeddings, models, etc.).

Setup

1) Ensure `ip.txt` in repo root contains a list of backends, one per line. Examples:

   - With explicit ports
     10.0.0.11:9001
     10.0.0.12:9002

   - Bare hosts (default port assumed via `BACKEND_DEFAULT_PORT`, default 8101)
     10.0.0.11
     10.0.0.12

   - Full URLs are also accepted
     http://10.0.0.11:8101/v1
     http://10.0.0.12:8101/v1

2) Install dependencies:
   pip install -r requirements.txt

3) Run the forwarder (default port 9000):
   uvicorn vllm_forwarder.app:app --host 0.0.0.0 --port 9000

Environment variables:
- IP_LIST_FILE (default: ip.txt)
- CHECK_INTERVAL_SECONDS (default: 10)
- BACKEND_DEFAULT_PORT (default: 8101; used only if an entry in `ip.txt` has no explicit port)
- FORCED_MODEL_NAME (optional; if set, overrides the `model` field in JSON POST requests like chat/completions, completions, embeddings)

Status endpoint:
- GET /_backends shows current backend health, running/waiting counts.

Using with OpenAI client (Python)

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",  # forwarder address
    api_key="EMPTY"  # if your backend doesn’t require auth
)

resp = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.2,
)
print(resp.choices[0].message)

For streaming:

from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="EMPTY")
with client.chat.completions.stream(
    model="default",
    messages=[{"role": "user", "content": "Stream a reply."}],
) as stream:
    for event in stream:
        if event.type == "chunk":
            print(event.delta, end="")

Notes
- The forwarder assumes all backends expose OpenAI-compatible endpoints under `/v1` and Prometheus metrics under `/metrics` (at the root i.e., remove `/v1` then append `/metrics`).
- If no backend is healthy, the forwarder responds with 503.
- If `FORCED_MODEL_NAME` is set (e.g., `FORCED_MODEL_NAME=grader`), the forwarder rewrites the `model` field in JSON POST bodies to that value; otherwise it proxies the body unchanged.
