# 待完成
## if use vllm_forwarder
```
cd vllm_fowarder
# change ip.txt
nohup uvicorn vllm_forwarder.app:app --host 0.0.0.0 --port 9099 --workers 4 > ../log/forwarder.log 2>&1 &
```