import time
from huggingface_hub import snapshot_download

# Model configuration
model_id = "Qwen/Qwen2.5-3B"
local_dir = "model_weight/Qwen/Qwen2.5-3B"

print("=" * 50)
print(f"Downloading model: {model_id}")
print(f"Destination: {local_dir}")
print("=" * 50)

max_retries = 20
retry_count = 0

while retry_count < max_retries:
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )

        print(f"\nDownload success! Saved to: {path}")
        break

    except Exception as e:
        retry_count += 1
        print(f"\nInterrupted ({retry_count}/{max_retries}): {e}")

        if retry_count < max_retries:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("\nDownload failed: max retries reached.")
