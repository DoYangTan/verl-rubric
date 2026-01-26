#!/bin/bash
# ================================================================
# GPU 实时监控脚本 (FULL 版)
# - 显示 GPU 名称、显存用量、GPU 利用率
# - 列出每个进程的用户、PID、显存、命令
# - 无闪屏：信息采集完再统一刷新
# ================================================================

REFRESH_INTERVAL=3  # 每几秒刷新一次

while true; do
    tmpfile=$(mktemp)

    {
        echo "===================== GPU 实时占用监控 (FULL) ====================="
        date "+更新时间: %Y-%m-%d %H:%M:%S"
        echo "------------------------------------------------------------------"

        # 获取 GPU 基本信息
        mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
        mapfile -t gpu_mem_total < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        mapfile -t gpu_mem_used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        mapfile -t gpu_util < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

        gpu_count=${#gpu_names[@]}

        echo "GPU 总览:"
        for ((i=0; i<$gpu_count; i++)); do
            name="${gpu_names[$i]}"
            total="${gpu_mem_total[$i]}"
            used="${gpu_mem_used[$i]}"
            util="${gpu_util[$i]}"
            ratio=$(awk "BEGIN {printf \"%.1f\", ($used/$total)*100}")
            printf "  GPU:%-2d %-25s  已用: %6s / %6s MiB (%.1f%%) | GPU 利用率: %3s%%\n" \
                   "$i" "$name" "$used" "$total" "$ratio" "$util"
        done

        echo "------------------------------------------------------------------"
        printf "%-4s | %-8s | %-8s | %-8s | %s\n" "GPU" "USER" "PID" "MEM(MiB)" "COMMAND"
        echo "------------------------------------------------------------------"

        # 获取所有 GPU 进程行（PID、GPU、MEM）
        mapfile -t gpuinfo < <(nvidia-smi | awk '/ C | G / {print $2, $5, $(NF-1)}')

        for line in "${gpuinfo[@]}"; do
            gpu=$(echo "$line" | awk '{print $1}')
            pid=$(echo "$line" | awk '{print $2}')
            mem=$(echo "$line" | awk '{print $3}')

            [[ -z "$pid" || "$pid" == "-" ]] && continue

            # 获取 ps 输出
            if psout=$(ps -o user=,cmd= -p "$pid" 2>/dev/null); then
                user=$(echo "$psout" | awk '{print $1}')
                cmd=$(echo "$psout" | cut -d' ' -f2- | cut -c1-80)
                printf "%-4s | %-8s | %-8s | %-8s | %s\n" "$gpu" "$user" "$pid" "$mem" "$cmd"
            fi
        done

        echo "------------------------------------------------------------------"
        echo "按 Ctrl+C 退出 | 刷新间隔: ${REFRESH_INTERVAL}s"
    } > "$tmpfile"

    clear
    cat "$tmpfile"
    rm -f "$tmpfile"

    sleep $REFRESH_INTERVAL
done
