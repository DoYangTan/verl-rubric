#!/bin/bash

# ===== 配置参数 =====
TP_SIZE=8   # 可选 1 / 2 / 4 / 8
BASE_PORT=8101

# 模型路径列表（根据TP_SIZE自动裁剪）
MODEL_PATHS=(
/lpai/dataset/model/0-1-0/model/Qwen3-235B-A22B-Thinking-2507
)

# ===== 脚本逻辑 =====
NUM_GPUS=8
MAX_MODELS=$((NUM_GPUS / TP_SIZE))

# 只取前 MAX_MODELS 个模型
MODEL_PATHS=("${MODEL_PATHS[@]:0:$MAX_MODELS}")

# 先停止已有的vllm服务
pkill -f "vllm serve"

# 启动每个模型的vllm服务
for i in "${!MODEL_PATHS[@]}"; do
    PORT=$((BASE_PORT + i))
    GPU_START=$((i * TP_SIZE))
    GPU_END=$((GPU_START + TP_SIZE - 1))
    
    # 拼接 GPU id 列表
    GPUS=$(seq -s, $GPU_START $GPU_END)
    MODEL_PATH="${MODEL_PATHS[$i]}"
    
    echo "启动模型服务: $MODEL_PATH 端口: $PORT 使用GPU: $GPUS (TP=$TP_SIZE)"
    
    CUDA_VISIBLE_DEVICES=$GPUS vllm serve "$MODEL_PATH" \
        --port $PORT \
        --host 0.0.0.0 \
        --tensor-parallel-size $TP_SIZE \
        --served-model-name default \ &

done

echo "所有vllm服务已启动 (TP=$TP_SIZE)"

        #--disable-log-requests \