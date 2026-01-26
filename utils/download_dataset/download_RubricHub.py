import sys
import time
from huggingface_hub import snapshot_download

dataset_id = "sojuL/RubricHub_v1"
local_dir = "data/RubricHub_v1/RuRL"
sub_folder_pattern = "RuRL/*"

print("="*50)
print(f"Downloading dataset: {dataset_id}")
print(f"Sub-folder pattern: {sub_folder_pattern}")
print(f"Destination: {local_dir}")
print("="*50)

max_retries = 10
retry_count = 0

while retry_count < max_retries:
    try:
        path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
            allow_patterns=[sub_folder_pattern]
        )
        
        print(f"\nDataset downloaded successfully!")
        print(f"Stored at: {path}")
        print("-" * 50)
        break

    except Exception as e:
        retry_count += 1
        print(f"\nDownload interrupted (Attempt {retry_count}/{max_retries}): {e}")
        print("Waiting 5 seconds before retry...")
        time.sleep(5)

if retry_count >= max_retries:
    print("\nMax retries reached. Download failed.")