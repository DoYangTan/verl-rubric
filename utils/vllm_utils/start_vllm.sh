#!/bin/bash

GPU_ID=8
PORT=8002
MODEL_PATH="model_weight/openai/gpt-oss-20b"
SERVED_NAME="gpt-oss-20b"
GPU_UTIL=0.95
TP_SIZE=1
LOG_FILE="vllm_server_${PORT}.log"

unset http_proxy https_proxy all_proxy
echo "Proxy settings cleared."
echo "Starting vllm service..."
echo "Model: $SERVED_NAME"
echo "Port: $PORT"
echo "GPU: $GPU_ID"
echo "Log file: $LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve $MODEL_PATH \
    --served-model-name $SERVED_NAME \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --gpu-memory-utilization $GPU_UTIL \
    > $LOG_FILE 2>&1 &

PID=$!
echo "Service started successfully! Background PID: $PID"
echo "You can view the logs using the following command:"
echo "tail -f $LOG_FILE"