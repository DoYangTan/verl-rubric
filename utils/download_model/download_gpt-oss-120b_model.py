import time
from huggingface_hub import snapshot_download

# Model configuration
model_id = "openai/gpt-oss-120b"
local_dir = "model_weight/openai/gpt-oss-120b"

print("="*50)
print(f"Downloading model: {model_id}")
print(f"Destination: {local_dir}")
print("WARNING: 120B model is very large; ensure you have enough disk space.")
print("="*50)

max_retries = 50  
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
        
        print("\nDownload success!")
        print(f"Saved to: {path}")
        break

    except Exception as e:
        retry_count += 1
        print(f"\nInterrupted ({retry_count}/{max_retries}): {e}")
        print("Retrying in 10 seconds...")
        time.sleep(10)

if retry_count >= max_retries:
    print("\nDownload failed: max retries reached. Please check network and disk space.")
