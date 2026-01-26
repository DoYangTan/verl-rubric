import torch
import time
import subprocess
import multiprocessing as mp

# === 可调参数 ===
THRESHOLD = 50          # 率阈值 (%)
CHECK_INTERVAL = 1    # 每次检测间隔 (秒)
RUN_SECONDS = 10        # run时间 (秒)

def get_gpu_utilization():
    """返回每张GPU的利用率列表"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        utils = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utils
    except Exception as e:
        print(f"[Error] 无法获取GPU利用率: {e}")
        return []

def gpu_task(gpu_id, tensor_size_mb=1, run_seconds=RUN_SECONDS):
    """单GPU占用任务，持续运行 run_seconds 秒"""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    elem_count = tensor_size_mb * 1024 * 1024 // 2
    matrix_side = int(elem_count ** 0.62)
    data_a = torch.randn(matrix_side, matrix_side, dtype=torch.float16, device=device)
    data_b = torch.randn(matrix_side, matrix_side, dtype=torch.float16, device=device)

    end_time = time.time() + run_seconds
    while time.time() < end_time:
        result = torch.matmul(data_a, data_b)
        data_a = result.detach()
        torch.cuda.synchronize()

    del data_a, data_b
    torch.cuda.empty_cache()

def monitor_and_run(threshold=THRESHOLD, check_interval=CHECK_INTERVAL, run_seconds=RUN_SECONDS):
    """监控 GPU 利用率，小于 threshold 时占用 GPU"""

    while True:
        utils = get_gpu_utilization()
        if not utils:
            time.sleep(check_interval)
            continue

        print("当前GPU利用率:", utils)

        low_util_gpus = [i for i, u in enumerate(utils) if u < threshold]
        if low_util_gpus:
            procs = [mp.Process(target=gpu_task, args=(i, 1, run_seconds)) for i in low_util_gpus]
            for p in procs:
                p.start()
            for p in procs:
                p.join()

        time.sleep(check_interval)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    monitor_and_run()