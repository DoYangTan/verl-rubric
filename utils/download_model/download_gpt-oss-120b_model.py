import sys
import os
import time
from huggingface_hub import snapshot_download

proxy_url = "http://127.0.0.1:10086"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
print(f"🌍 已设置代理: {proxy_url}")
# ===========================================

model_id = "openai/gpt-oss-120b"  # <--- 修改这里
local_dir = "model_weight/openai/gpt-oss-120b" # <--- 修改这里，建议和模型名保持一致
# ----------------

print("="*50)
print(f"正在尝试下载模型: {model_id}")
print(f"目标路径: {local_dir}")
print("⚠️ 警告: 120B 模型体积巨大(约240GB)，请确保磁盘空间充足！")
print("="*50)

max_retries = 50  
retry_count = 0

while retry_count < max_retries:
    try:
        # 执行下载
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,  
            
            max_workers=4          
        )
        
        print(f"\n✅ 下载成功！")
        print(f"模型保存在: {path}")
        break  # 下载成功，跳出循环

    except Exception as e:
        retry_count += 1
        print(f"\n❌ 下载中断 (尝试 {retry_count}/{max_retries}): {e}")
        print("⏳ 网络不稳定/文件过大，等待 10 秒后自动重试...")
        time.sleep(10)  # 120B 下载失败后建议多冷却一会儿

if retry_count >= max_retries:
    print("\n❌ 达到最大重试次数，下载失败。请检查网络或磁盘空间。")