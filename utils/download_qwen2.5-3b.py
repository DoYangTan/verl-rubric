import os
import time
from huggingface_hub import snapshot_download

# ================= 1. 代理配置 =================
# 设置本地代理端口 10086
proxy_url = "http://127.0.0.1:10086"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
print(f"🌍 已配置代理: {proxy_url}")
# ==============================================

# ================= 2. 模型参数配置 =================
# 更改为 Qwen2.5-3B
model_id = "Qwen/Qwen2.5-3B"
# 更新本地保存路径以匹配模型名称
local_dir = "model_weight/Qwen/Qwen2.5-3B"
# ====================================================

print("="*50)
print(f"📦 正在下载模型: {model_id}")
print(f"📂 目标保存路径: {local_dir}")
print("="*50)

max_retries = 20
retry_count = 0

while retry_count < max_retries:
    try:
        # 执行下载
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 下载真实文件
            resume_download=True,         # 支持断点续传
            max_workers=4                 # 限制并发，保持稳定
        )
        
        print(f"\n✅ 下载成功！模型已保存至: {path}")
        break 

    except Exception as e:
        retry_count += 1
        print(f"\n❌ 下载中断 (尝试 {retry_count}/{max_retries}): {e}")
        
        if retry_count < max_retries:
            print("⏳ 网络不稳定，5秒后重试...")
            time.sleep(5)
        else:
            print("\n❌ 下载失败：已达到最大重试次数。")