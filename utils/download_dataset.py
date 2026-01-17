import sys
import os
import time
from huggingface_hub import snapshot_download

# ================= 1. 代理配置 (如果需要) =================
# 如果你的服务器网络能直连 HF，可以注释掉下面三行
proxy_url = "http://127.0.0.1:10086"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
print(f"🌍 已设置代理: {proxy_url}")
# ========================================================

# ================= 2. 参数配置 =================
# 仓库ID (去掉了 tree/main/... 后面的部分)
dataset_id = "sojuL/RubricHub_v1"

# 设置本地保存路径
# 注意：因为只下载特定子文件夹，本地路径最好也对应清楚
local_dir = "data/RubricHub_v1/RuRL"

# 指定子文件夹路径 (只下载 RubricHub_v1/RuRL 下的内容)
sub_folder_pattern = "RubricHub_v1/RuRL/*" 
# ========================================================

print("="*50)
print(f"📦 正在尝试下载数据集: {dataset_id}")
print(f"🎯 指定下载子目录: {sub_folder_pattern}")
print(f"📂 目标保存路径: {local_dir}")
print("="*50)

# === 下载逻辑 ===
max_retries = 10
retry_count = 0

while retry_count < max_retries:
    try:
        # 执行下载
        path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",      # <--- 关键点：必须指定为 dataset
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 下载真实文件，而不是快捷方式
            resume_download=True,     # 支持断点续传
            max_workers=4,            # 限制并发数防止报错
            allow_patterns=[sub_folder_pattern] # <--- 新增：只下载指定文件夹的内容
        )
        
        print(f"\n✅ 数据集下载成功！")
        print(f"存储路径: {path}")
        print("-" * 50)
        print("💡 使用提示: 数据可能位于子文件夹中，加载时请注意路径:")
        print(f"dataset = load_dataset('parquet', data_files='{local_dir}/{sub_folder_pattern.replace('*', '*.parquet')}')")
        break

    except Exception as e:
        retry_count += 1
        print(f"\n❌ 下载中断 (尝试 {retry_count}/{max_retries}): {e}")
        print("⏳ 等待 5 秒后重试...")
        time.sleep(5)

if retry_count >= max_retries:
    print("\n❌ 达到最大重试次数，下载失败。")