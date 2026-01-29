import os
import shutil

# ================= CONFIGURATION =================
# Target directory path
TARGET_DIR = "/workspace/verl-rubric-dev/checkpoints/rubrichub_v1_Medical/baseline_PPO_qwen2.5-3b_oss-20b"

# Threshold: Steps less than or equal to this number will be deleted
THRESHOLD_STEP = 700

# SAFETY FLAG: Set to True to test, Set to False to actually delete
DRY_RUN = False
# =================================================

def main():
    # Verify directory exists
    if not os.path.exists(TARGET_DIR):
        print(f"[Error] Directory not found: {TARGET_DIR}")
        return

    print(f"Scanning directory: {TARGET_DIR}")
    print(f"Targeting checkpoints with global_step <= {THRESHOLD_STEP}")
    
    if DRY_RUN:
        print(">>> DRY RUN MODE: No files will be deleted. <<<")
    else:
        print(">>> WARNING: REAL DELETION MODE ENABLED. <<<")

    deleted_count = 0
    
    # List all items in the directory
    try:
        items = os.listdir(TARGET_DIR)
    except Exception as e:
        print(f"[Error] Could not list directory: {e}")
        return

    for item in items:
        full_path = os.path.join(TARGET_DIR, item)
        
        # Check if it is a directory and matches the naming pattern
        if os.path.isdir(full_path) and item.startswith("global_step_"):
            try:
                # Extract the step number (e.g., global_step_100 -> 100)
                step_str = item.split('_')[-1]
                step_num = int(step_str)
                
                # Compare step number
                if step_num <= THRESHOLD_STEP:
                    if DRY_RUN:
                        print(f"[Would Delete] {item} (Step {step_num})")
                    else:
                        print(f"[Deleting] {item}...")
                        shutil.rmtree(full_path) # Recursively delete directory
                    
                    deleted_count += 1
            except ValueError:
                print(f"[Skipping] Could not parse step number from: {item}")
                continue

    print("-" * 30)
    if DRY_RUN:
        print(f"Dry run complete. Found {deleted_count} directories to delete.")
        print("To execute, change 'DRY_RUN = True' to 'False' in the script.")
    else:
        print(f"Operation complete. Deleted {deleted_count} directories.")
        print("Please run 'df -h' to verify freed space.")

if __name__ == "__main__":
    main()