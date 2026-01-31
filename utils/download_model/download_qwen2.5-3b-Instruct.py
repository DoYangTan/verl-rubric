import time
from huggingface_hub import snapshot_download

# Model configuration (Instruct)
model_id = "Qwen/Qwen2.5-3B-Instruct"
local_dir = "model_weight/Qwen/Qwen2.5-3B-Instruct"

print(f"Downloading: {model_id} to {local_dir} ...")

max_retries = 20
retry_count = 0

while retry_count < max_retries:
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        print(f"\nSuccess! Model saved at: {path}")
        break 

    except Exception as e:
        retry_count += 1
        print(f"\nError ({retry_count}/{max_retries}): {e}")
        
        if retry_count < max_retries:
            time.sleep(5)
        else:
            print("\nDownload failed.")
