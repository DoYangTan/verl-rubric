pip install -r requirements.txt
uvicorn vllm_forwarder.app:app --host 0.0.0.0 --port 9000 --workers 4