# Verl-Rubric
## prepare dataset
```
# RubricHub
python utils/download_dataset/download_RubricHub.py
# Healthbench
python utils/process_dataset/prepare_healthbench.py
```
## download model
```
# gpt-oss-20b
python utils/download_model/download_gpt-oss-20b_model.py
# gpt-oss-120b
python utils/download_model/download_gpt-oss-120b_model.py
# Qwen2.5-3B-Instruct
python utils/download_model/download_qwen2.5-3b-Instruct.py
# Qwen2.5-3B
python utils/download_model/download_qwen2.5-3b.py
# Qwen2.5-7B-Instruct
python utils/download_model/download_qwen2.5-7b-Instruct.py
```
## vllm_server
```
# Please modify lines 3-9 of the utils/vllm_utils/start_vllm.sh script before running it.
python utils/vllm_utils/start_vllm.sh
```
## [Optional]To run multiple vLLM instances and set up a single forwarding port, please follow the steps below.
```
cd vllm_fowarder
# change ip.txt
nohup uvicorn vllm_forwarder.app:app --host 0.0.0.0 --port 9099 --workers 4 > ../log/forwarder.log 2>&1 &
```
## 解耦val和train阶段的VLLM_TEMPERATURE
```
# .env 保证val-score step0一致
VLLM_TEMPERATURE=1.0
VLLM_TEMPERATURE_VAL=0
```