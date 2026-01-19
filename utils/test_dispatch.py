import openai
import concurrent.futures
import time

# 配置：指向你的 Forwarder 端口 9099
client = openai.OpenAI(
    base_url="http://localhost:9099/v1",
    api_key="EMPTY",
)

# 你的模型名称
MODEL_NAME = "gpt-oss-120b" 

def send_request(idx):
    try:
        start_time = time.time()
        
        # --- 修改 1: 优化 Prompt，尝试抑制过长的推理 ---
        prompt_content = "Hi. Answer briefly."
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_content}
            ],
            # --- 修改 2: 关键！必须增加 Token 数，否则推理模型还没想完就被截断了 ---
            max_tokens=200, 
            temperature=0.7,
        )
        duration = time.time() - start_time
        
        # 获取返回内容
        message = completion.choices[0].message
        content = message.content
        
        # --- 修改 3: 如果 content 为空，尝试读取 reasoning_content (推理内容) ---
        # 很多新模型把思维链放在 reasoning_content 字段里
        if not content and hasattr(message, 'reasoning_content') and message.reasoning_content:
            final_output = f"[思考过程] {message.reasoning_content[:50]}..."
            status_icon = "🧠" # 表示返回的是思考过程
        elif content:
            final_output = content.strip()
            status_icon = "✅"
        else:
            final_output = "❌ 空内容 (Token可能不足)"
            status_icon = "⚠️"

        print(f"{status_icon} 请求 #{idx} 完成! 耗时: {duration:.2f}s | 回复: {final_output}")
        return True
        
    except Exception as e:
        print(f"❌ 请求 #{idx} 失败: {e}")
        return False

if __name__ == "__main__":
    total_requests = 20
    concurrency = 5

    print(f"🚀 开始测试: 向端口 9099 发送 {total_requests} 个请求 (并发数: {concurrency})...")
    
    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        concurrent.futures.wait(futures)
    
    print(f"\n🏁 测试结束! 总耗时: {time.time() - start_all:.2f}s")