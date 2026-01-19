pip install -r requirements.txt
uvicorn vllm_forwarder.app:app --host :: --port 8000 --workers 4