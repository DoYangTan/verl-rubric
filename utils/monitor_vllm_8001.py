import time
import requests
import os
import json
from datetime import datetime

# ================= 配置区域 =================
# 你的 vLLM 服务地址
BASE_URL = "http://localhost:8001"
# 强制设置不走代理（这是连通的关键！）
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
# ===========================================

def get_colored(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "cyan": "\033[96m", "reset": "\033[0m"}
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def get_vllm_metrics():
    """获取 vLLM 内部负载状态"""
    try:
        # vLLM 默认会在 /metrics 暴露 Prometheus 格式指标
        resp = requests.get(f"{BASE_URL}/metrics", timeout=1)
        if resp.status_code != 200:
            return None
        
        metrics = {}
        for line in resp.text.split('\n'):
            if line.startswith("vllm:num_requests_running"):
                metrics['running'] = float(line.split()[-1])
            elif line.startswith("vllm:num_requests_waiting"):
                metrics['waiting'] = float(line.split()[-1])
        return metrics
    except:
        return None

def test_inference(model_name):
    """发送真实请求测试模型反应"""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        # 建议调大 max_tokens，5太小了容易导致截断或无内容
        "max_tokens": 50, 
        "temperature": 0.1
    }
    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=10) # 稍微增加一点超时时间
        latency = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            # === 修改开始：安全获取 content ===
            try:
                message = data['choices'][0]['message']
                content = message.get('content')
                
                # 如果 content 是 None (例如只输出了 reasoning_content 或纯工具调用)
                if content is None:
                    # 尝试获取推理内容 (针对 R1/推理类模型)
                    reasoning = message.get('reasoning_content', '')
                    if reasoning:
                        res_text = f"[正在思考] {reasoning[:20]}..."
                    else:
                        res_text = "<返回内容为空>"
                else:
                    res_text = content.strip()
                    if not res_text:
                        res_text = "<返回空白字符>"
            except Exception as parse_err:
                 return True, f"解析异常: {str(parse_err)} ({latency:.1f}ms)"
            # === 修改结束 ===

            return True, f"{res_text} ({latency:.1f}ms)"
        else:
            return False, f"HTTP Error {resp.status_code}"
    except Exception as e:
        return False, str(e)
    """发送真实请求测试模型反应"""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "temperature": 0.1
    }
    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=5)
        latency = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            res_text = resp.json()['choices'][0]['message']['content'].strip()
            return True, f"{res_text} ({latency:.1f}ms)"
        else:
            return False, f"HTTP Error {resp.status_code}"
    except Exception as e:
        return False, str(e)

def get_model_name():
    """自动获取模型名称"""
    try:
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=2)
        return resp.json()['data'][0]['id']
    except:
        return "model_weight/openai/gpt-oss-120b" # 兜底

def main():
    print(get_colored(f"🚀 开始监控 vLLM 端口: {BASE_URL}", "cyan"))
    print(f"代理状态: no_proxy={os.environ.get('no_proxy')}")
    print("-" * 60)
    
    # 1. 先获取一次模型名
    model_name = get_model_name()
    print(f"目标模型: {model_name}\n")

    counter = 0
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 2. 获取负载指标 (这是最重要的监控！)
        metrics = get_vllm_metrics()
        
        if metrics is None:
            status = get_colored("连接失败 (Offline)", "red")
            detail = "请检查 1.端口是否开启 2.防火墙"
        else:
            # 根据负载变色
            r_color = "green" if metrics.get('running', 0) > 0 else "yellow"
            status = get_colored("服务在线 (Online)", "green")
            detail = f"正在处理: {get_colored(int(metrics.get('running', 0)), r_color)} | 排队中: {int(metrics.get('waiting', 0))}"

        print(f"[{timestamp}] {status} | {detail}", end="\r")

        # 3. 每 5 秒发一次测试请求，证明它能说话
        if counter % 5 == 0 and metrics is not None:
            print() # 换行防止覆盖
            success, msg = test_inference(model_name)
            if success:
                print(f"   └── [测试生成] 成功: {get_colored(msg, 'cyan')}")
            else:
                print(f"   └── [测试生成] 失败: {get_colored(msg, 'red')}")
            print("-" * 60)
        
        counter += 1
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n监控结束")