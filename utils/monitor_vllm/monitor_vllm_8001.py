import time
import requests
import json
from datetime import datetime

# ================= Configuration =================
# Your vLLM server address
BASE_URL = "http://localhost:8001"
# Proxy settings have been removed as requested
# ===============================================

def get_colored(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "cyan": "\033[96m", "reset": "\033[0m"}
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def get_vllm_metrics():
    """Get vLLM internal load status"""
    try:
        # vLLM exposes Prometheus metrics at /metrics by default
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
    """Send real request to test model response"""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        # Suggest increasing max_tokens to avoid truncation
        "max_tokens": 200, 
        "temperature": 0.1
    }
    try:
        start = time.time()
        # Increased timeout slightly for stability
        resp = requests.post(url, json=payload, timeout=10) 
        latency = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            # === Start: Safely retrieve content ===
            try:
                message = data['choices'][0]['message']
                content = message.get('content')
                
                # If content is None (e.g., only reasoning_content or tool calls)
                if content is None:
                    # Try getting reasoning content (for R1/reasoning models)
                    reasoning = message.get('reasoning_content', '')
                    if reasoning:
                        res_text = f"[Thinking] {reasoning[:20]}..."
                    else:
                        res_text = "<Empty Content>"
                else:
                    res_text = content.strip()
                    if not res_text:
                        res_text = "<Whitespace Content>"
            except Exception as parse_err:
                 return True, f"Parse Error: {str(parse_err)} ({latency:.1f}ms)"
            # === End ===

            return True, f"{res_text} ({latency:.1f}ms)"
        else:
            return False, f"HTTP Error {resp.status_code}"
    except Exception as e:
        return False, str(e)

def get_model_name():
    """Automatically retrieve model name"""
    try:
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=2)
        return resp.json()['data'][0]['id']
    except:
        return "model_weight/openai/gpt-oss-120b" # Fallback

def main():
    print(get_colored(f"🚀 Starting vLLM monitoring: {BASE_URL}", "cyan"))
    print("-" * 60)
    
    # 1. Get model name once first
    model_name = get_model_name()
    print(f"Target Model: {model_name}\n")

    counter = 0
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 2. Get load metrics (Critical monitoring!)
        metrics = get_vllm_metrics()
        
        if metrics is None:
            status = get_colored("Connection Failed (Offline)", "red")
            detail = "Check: 1. Port open 2. Firewall"
        else:
            # Color code based on load
            r_color = "green" if metrics.get('running', 0) > 0 else "yellow"
            status = get_colored("Service Online", "green")
            detail = f"Running: {get_colored(int(metrics.get('running', 0)), r_color)} | Waiting: {int(metrics.get('waiting', 0))}"

        print(f"[{timestamp}] {status} | {detail}", end="\r")

        # 3. Send a test request every 5 seconds to verify functionality
        if counter % 5 == 0 and metrics is not None:
            print() # New line to avoid overwriting
            success, msg = test_inference(model_name)
            if success:
                print(f"   └── [Test Gen] Success: {get_colored(msg, 'cyan')}")
            else:
                print(f"   └── [Test Gen] Failed: {get_colored(msg, 'red')}")
            print("-" * 60)
        
        counter += 1
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped")