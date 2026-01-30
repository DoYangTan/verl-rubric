#!/bin/bash
# ================================================================
# GPU 实时监控脚本 (终极版：动态时钟补偿 + 极速查询)
# ================================================================

REFRESH_INTERVAL=1

# --- 颜色定义 ---
ESC=$(printf '\033')
RESET="${ESC}[0m"
BOLD="${ESC}[1m"
RED="${ESC}[31m"
GREEN="${ESC}[32m"
YELLOW="${ESC}[33m"
BLUE="${ESC}[34m"
MAGENTA="${ESC}[35m"
CYAN="${ESC}[36m"
WHITE="${ESC}[37m"
GREY="${ESC}[90m"

USER_COLORS=($RED $GREEN $YELLOW $BLUE $MAGENTA $CYAN)
BAR_CHAR="█"
EMPTY_CHAR="░"

if ((BASH_VERSINFO[0] < 4)); then echo "Error: Bash 4.0+ required."; exit 1; fi
trap 'tput cnorm; echo -e "${RESET}"; exit 0' INT TERM
tput civis

# --- 绘图函数 ---
draw_bar() {
    local current=$1; local total=$2; local width=$3; local use_gradient=$4
    local percent=0
    if [ "$total" -gt 0 ]; then percent=$((current * 100 / total)); fi

    local color=$GREEN
    if [ "$use_gradient" == "yes" ]; then
        if [ "$percent" -ge 80 ]; then color=$RED
        elif [ "$percent" -ge 50 ]; then color=$YELLOW; fi
    else
        color=$BLUE
    fi
    
    if [ "$width" -lt 0 ]; then width=0; fi
    local fill_len=$((percent * width / 100))
    if [ "$fill_len" -lt 0 ]; then fill_len=0; fi
    if [ "$fill_len" -gt "$width" ]; then fill_len=$width; fi
    local empty_len=$((width - fill_len))
    
    local bar=""; for ((j=0; j<fill_len; j++)); do bar+="${BAR_CHAR}"; done
    local empty=""; for ((j=0; j<empty_len; j++)); do empty+="${EMPTY_CHAR}"; done
    printf "${color}%s${GREY}%s${RESET}" "$bar" "$empty"
}

get_user_color_dynamic() {
    local user_str=$1; local sum=0
    if [ -z "$user_str" ] || [ "$user_str" == "?" ]; then echo "$WHITE"; return; fi
    for (( i=0; i<${#user_str}; i++ )); do
        local char=$(printf "%d" "'${user_str:$i:1}")
        sum=$((sum + char))
    done
    local idx=$((sum % ${#USER_COLORS[@]}))
    echo "${USER_COLORS[$idx]}"
}

# --- 预先建立 GPU UUID 到 Index 的映射 (只跑一次，极大加速) ---
# nvidia-smi 查询进程时返回的是 UUID 或 BusID，我们需要映射回 0,1,2
declare -A GPU_UUID_TO_ID
while IFS=, read -r idx uuid; do
    idx=$(echo "$idx" | xargs)
    uuid=$(echo "$uuid" | xargs)
    GPU_UUID_TO_ID["$uuid"]="$idx"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

while true; do
    # [计时开始] 记录循环开始的纳秒时间
    start_time_ns=$(date +%s%N)
    
    # 1. 基础数据准备
    term_width=$(tput cols)
    if [ -z "$term_width" ] || [ "$term_width" -lt 70 ]; then term_width=70; fi
    current_time=$(date "+%Y-%m-%d %H:%M:%S")
    buffer=""

    # 2. 面板头部
    buffer+="${BOLD}========================== GPU 实时监控面板 ==========================${RESET}\n"
    # 这里显示的刷新率还是预设值，但实际体验会非常接近这个值
    info_line=$(printf " 时间: ${YELLOW}%s${RESET} | 刷新: ${CYAN}%ss${RESET} | 终端: ${WHITE}%d cols${RESET}" \
        "$current_time" "$REFRESH_INTERVAL" "$term_width")
    buffer+=" $info_line\n"
    buffer+="${GREY}----------------------------------------------------------------------${RESET}\n"

    # --- GPU 区域宽度计算 ---
    avail_bar_space=$((term_width - 54))
    if [ "$avail_bar_space" -lt 2 ]; then avail_bar_space=2; fi
    w_bar_vram=$((avail_bar_space * 60 / 100))
    w_bar_util=$((avail_bar_space - w_bar_vram))
    w_col_vram_total=$((2 + w_bar_vram + 1 + 15))
    w_col_util_total=$((2 + w_bar_util + 1 + 4))

    # 动态表头
    header_line=$(printf "${BOLD}%-3s %-18s | %-*.*s | %-*.*s${RESET}" \
        "ID" "MODEL" "$w_col_vram_total" "$w_col_vram_total" "VRAM USAGE" "$w_col_util_total" "$w_col_util_total" "UTIL")
    buffer+="${header_line}\n"
    
    dash_vram=$(printf '%*s' "$w_col_vram_total" "" | tr ' ' '-')
    dash_util=$(printf '%*s' "$w_col_util_total" "" | tr ' ' '-')
    buffer+="${GREY}--- ------------------   $dash_vram   $dash_util${RESET}\n"

    # 3. GPU 数据 (第一次 nvidia-smi 调用)
    gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)
    while IFS=, read -r idx name total used util; do
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs | cut -c1-18)
        total=$(echo "$total" | xargs)
        used=$(echo "$used" | xargs)
        util=$(echo "$util" | xargs)

        mem_bar=$(draw_bar "$used" "$total" "$w_bar_vram" "yes")
        util_bar=$(draw_bar "$util" 100 "$w_bar_util" "yes")
        mem_text=$(printf "%5s/%-5s MiB" "$used" "$total")
        util_text=$(printf "%3s%%" "$util")
        
        line_str=$(printf " %-3s %-18s | [%s] ${WHITE}%-15s${RESET} | [%s] ${WHITE}%-4s${RESET}" \
               "$idx" "$name" "$mem_bar" "$mem_text" "$util_bar" "$util_text")
        buffer+="$line_str\n"
    done <<< "$gpu_info"

    # 4. 进程列表
    buffer+="\n${GREY}----------------------------------------------------------------------${RESET}\n"

    w_proc_gpu=3; w_proc_user=10; w_proc_pid=8; w_proc_mem=10
    fixed_proc_occupied=36 
    w_cmd=$((term_width - fixed_proc_occupied - 1))
    if [ "$w_cmd" -lt 10 ]; then w_cmd=10; fi

    proc_header=$(printf "${BOLD} %-${w_proc_gpu}s %-${w_proc_user}s %-${w_proc_pid}s %-${w_proc_mem}s %-${w_cmd}s${RESET}" \
        "GPU" "USER" "PID" "MEM" "COMMAND")
    buffer+="${proc_header}\n"
    dash_cmd=$(printf '%*s' "$w_cmd" "" | tr ' ' '-')
    buffer+="${GREY} --- ---------- -------- ---------- $dash_cmd${RESET}\n"

    # === [极速优化部分] ===
    # 不再解析 nvidia-smi 的大表格文本，改为 CSV 查询
    # 速度提升约 5-10 倍
    mapfile -t proc_lines_csv < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)

    pids_to_query=""
    for line in "${proc_lines_csv[@]}"; do
        # CSV格式: GPU-xxxx, 12345, 4096
        # 需要用 IFS 拆分
        IFS=, read -r p_uuid p_pid p_mem <<< "$line"
        p_pid=$(echo "$p_pid" | xargs) # 去空格
        if [[ "$p_pid" =~ ^[0-9]+$ ]]; then
            pids_to_query+="$p_pid,"
        fi
    done
    pids_to_query=${pids_to_query%,} # 去尾部逗号

    declare -A map_user
    declare -A map_cmd
    
    if [ -n "$pids_to_query" ]; then
        while read -r arg_pid arg_user arg_cmd; do
            map_user["$arg_pid"]="$arg_user"
            map_cmd["$arg_pid"]="$arg_cmd"
        done < <(ps -p "$pids_to_query" -o pid=,user=,args= 2>/dev/null) 
    fi

    proc_count=0
    limit_lines=20 

    for line in "${proc_lines_csv[@]}"; do
        if [ "$proc_count" -ge "$limit_lines" ]; then break; fi

        IFS=, read -r p_uuid p_pid p_mem <<< "$line"
        p_uuid=$(echo "$p_uuid" | xargs)
        p_pid=$(echo "$p_pid" | xargs)
        p_mem=$(echo "$p_mem" | xargs)

        [[ "$p_pid" =~ ^[0-9]+$ ]] || continue

        # [查表] 将 UUID 转换为 ID (0, 1, 2...)
        p_gpu="${GPU_UUID_TO_ID[$p_uuid]}"
        # 如果有些驱动版本 compute-apps 返回的不是 uuid，这里做个保护
        if [ -z "$p_gpu" ]; then p_gpu="?"; fi
        
        # 加上 MiB 单位
        p_mem="${p_mem}MiB"

        p_user="${map_user[$p_pid]}"
        p_cmd_raw="${map_cmd[$p_pid]}"

        if [ -z "$p_user" ]; then 
            p_user="?" 
            p_cmd_raw="[Process info unavailable]"
        fi

        p_cmd_clean=$(echo "$p_cmd_raw" | tr '\n' ' ' | xargs)
        if [ ${#p_cmd_clean} -gt "$w_cmd" ]; then
            p_cmd_clean="${p_cmd_clean:0:$w_cmd}"
        fi

        u_color=$(get_user_color_dynamic "$p_user")
        
        row_fmt=" %-${w_proc_gpu}s ${u_color}%-${w_proc_user}.${w_proc_user}s${RESET} %-${w_proc_pid}s %-${w_proc_mem}s %-${w_cmd}s"
        line_content=$(printf "$row_fmt" "$p_gpu" "$p_user" "$p_pid" "$p_mem" "$p_cmd_clean")
        buffer+="${line_content}\n"
        ((proc_count++))
    done

    if [ "$proc_count" -eq 0 ]; then
        buffer+=" (无显存占用进程)\n"
    fi

    clear
    echo -e "$buffer"

    # === [动态时钟补偿核心] ===
    end_time_ns=$(date +%s%N)
    # 计算耗时 (纳秒)
    elapsed_ns=$((end_time_ns - start_time_ns))
    # 转换成秒 (浮点计算需要 python 或 bc，这里用 bash 整数运算近似处理)
    # 目标休眠毫秒数
    target_sleep_ms=$((REFRESH_INTERVAL * 1000))
    # 实际耗时毫秒数
    elapsed_ms=$((elapsed_ns / 1000000))
    # 剩余需要休眠的毫秒数
    sleep_ms=$((target_sleep_ms - elapsed_ms))
    
    # 如果耗时超过了设定间隔，则不休眠(立即刷新)，否则休眠剩余时间
    if [ "$sleep_ms" -gt 0 ]; then
        # 转换回秒 (比如 0.5 秒)
        sleep_sec=$(echo "scale=3; $sleep_ms / 1000" | bc 2>/dev/null)
        # 如果没有 bc 命令，使用 python 辅助计算，或者简单的整数 sleep
        if [ -z "$sleep_sec" ]; then
             # 降级方案：如果没有 bc，使用 python (你机器上有)
             sleep_sec=$(python3 -c "print($sleep_ms / 1000.0)")
        fi
        sleep "$sleep_sec"
    fi
done