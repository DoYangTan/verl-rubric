import os
import time
from huggingface_hub import snapshot_download

proxy_url = "http://127.0.0.1:10086"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
print(f"🌍proxy: {proxy_url}")

# Model configuration
model_id = "openai/gpt-oss-20b"
local_dir = "model_weight/openai/gpt-oss-20b"

print(f"Downloading {model_id} to {local_dir}...")

max_retries = 20
retry_count = 0

while retry_count < max_retries:
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4  # Reduced concurrency for stability
        )
        
        print(f"\nDownload success! Saved to: {path}")
        break 

    except Exception as e:
        retry_count += 1
        print(f"\nInterrupted ({retry_count}/{max_retries}): {e}")
        
        if retry_count < max_retries:
            print("Connection unstable. Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("\nDownload failed: Max retries reached.")