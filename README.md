# verl-rubric

> verl-rubric extends VERL to support rubric-based scoring (multi-criterion, point-weighted evaluation) for training and reproducible validation of LLMs.

## Getting Started 🚀

### Installation 🧰
We recommend Python **3.12**.

```bash
# 1) Create and activate a conda environment
conda create -n verl-rubric python=3.12 -y
conda activate verl-rubric

# 2) Install Python dependencies
python -m pip install -U pip
python -m pip install -r requirements.txt

# 3) (Recommended for development) Install this repo in editable mode
python -m pip install -e .
```

### Configuration ⚙️

#### CUDA (required) 🟦

This project expects **CUDA >= 12.8** (driver/runtime) for the GPU stack used by the training and inference components.

```bash
# Check driver / CUDA runtime version
nvidia-smi
```

If you manage CUDA via conda, you can install a compatible toolkit:

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

#### Environment variables 🔧

We use `.env` to configure both training and the LLM judge / vLLM endpoints. This repo does **not** require you to commit your own `.env`; instead, we provide a reference template in [`.env.example`](.env.example). Copy it to `.env` and edit it for your environment.

Key fields:

- vLLM / judge:
  - `VLLM_BASE_URL` (OpenAI-compatible base URL; must include `/v1`; supports comma-separated URLs)
  - `VLLM_MODEL`
  - `VLLM_TEMPERATURE` (judge temperature for training-time scoring)
  - `VLLM_TEMPERATURE_VAL` (judge temperature for validation; set `0` for determinism)
- GPUs:
  - `CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `GPU_MEMORY_UTILIZATION`
- Model + data:
  - `MODEL_PATH`, `MODEL_DTYPE`
  - `TRAIN_FILE`, `VAL_FILE`
  - `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`

Optional:
- `VAL_ONLY=True` to run validation only (scripts map it to `trainer.val_only`).
- `WANDB_API_KEY` to enable W&B logging (or remove it / disable W&B in your scripts).

### Data Preparation 📦

This project uses:

- **RubricHub v1 (RuRL)** as the training dataset (hosted on Hugging Face): `sojuL/RubricHub_v1`
- **HealthBench** as the validation dataset (processed locally into parquet)

#### Download RubricHub (RuRL) from Hugging Face 🤗

```bash
python3 utils/download_dataset/download_RubricHub.py
```

By default this downloads (and resumes if interrupted) into `data/RubricHub_v1/RuRL/` and only fetches the `RuRL/*` subset.

#### Prepare HealthBench parquet (from local JSONL) 🧾

Place the HealthBench JSONL files under `raw_data/healthbench/`:

- `raw_data/healthbench/healthbench_train.jsonl`
- `raw_data/healthbench/healthbench_eval.jsonl`

Then run:

```bash
python3 utils/process_dataset/prepare_healthbench.py \
  --local_dir raw_data/healthbench \
  --output_dir data/health_bench
```

This will generate:

- `data/health_bench/healthbench_train.parquet`
- `data/health_bench/healthbench_val.parquet`

### Start vLLM 🧠

Before starting vLLM, make sure you have downloaded the **grader model** you plan to serve.

Example (recommended default): **`openai/gpt-oss-20b`**

```bash
python3 utils/download_model/download_gpt-oss-20b_model.py
```

We provide additional download helpers under [`utils/download_model/`](utils/download_model/) (e.g. `openai/gpt-oss-120b`), but note that larger models may require substantial disk space.

We provide a helper script at [`utils/vllm_utils/start_vllm.sh`](utils/vllm_utils/start_vllm.sh) to start an OpenAI-compatible vLLM server.

#### Configure the launcher script 🛠️

Open [`utils/vllm_utils/start_vllm.sh`](utils/vllm_utils/start_vllm.sh) and edit these parameters:

- `GPU_ID`: which GPU to run vLLM on (mapped to `CUDA_VISIBLE_DEVICES`)
- `PORT`: vLLM HTTP port (your base URL becomes `http://localhost:${PORT}/v1`)
- `MODEL_PATH`: local model path to serve (e.g. under `model_weight/`)
- `SERVED_NAME`: model name exposed to clients (must match `VLLM_MODEL` in `.env` if you use the judge)
- `GPU_UTIL`: `--gpu-memory-utilization` (0–1)
- `TP_SIZE`: `--tensor-parallel-size`
- `LOG_FILE`: log path (defaults to `log/vllm_server_${PORT}.log`)

#### Start the server ▶️

```bash
bash utils/vllm_utils/start_vllm.sh
tail -f log/vllm_server_*.log
```

After the server is up, set your `.env` accordingly (example):

```bash
VLLM_BASE_URL=http://localhost:8002/v1
VLLM_MODEL=gpt-oss-20b
```

#### Optional: run multiple vLLM instances via a forwarder 🔀

If you want higher throughput (e.g., many concurrent judge/rollout requests) or want to utilize multiple GPUs, you can run **multiple vLLM servers** and route them behind a **single OpenAI-compatible endpoint** using the [`vllm_fowarder/`](vllm_fowarder/) service. This avoids changing client code/config when scaling from 1 → N backends, and provides simple load balancing.

1) Start multiple vLLM instances (one per GPU/port). For example, run [`utils/vllm_utils/start_vllm.sh`](utils/vllm_utils/start_vllm.sh) multiple times after changing `GPU_ID` / `PORT`.

2) Configure backends in [`vllm_fowarder/ip.txt`](vllm_fowarder/ip.txt) (one per line; **do not** include `/v1`):

```text
http://localhost:8001
http://localhost:8002
```

3) Start the forwarder:

```bash
cd vllm_fowarder
nohup uvicorn vllm_forwarder.app:app --host 0.0.0.0 --port 9099 --workers 4 > ../log/forwarder.log 2>&1 &
```

4) Point clients to the forwarder:

```bash
VLLM_BASE_URL=http://localhost:9099/v1
```

### Run Training 🏃

Before training, make sure you have downloaded the **actor/base model** to `model_weight/` (example: Qwen2.5-7B-Instruct):

```bash
python3 utils/download_model/download_qwen2.5-7b-Instruct.py
```

Then update your `.env` to point `MODEL_PATH` to the downloaded checkpoint (and set `MODEL_DTYPE`, GPU settings, and dataset paths as needed).

We provide reference scripts for **PPO / GRPO / GDPO** under [`rubric_scripts/`](rubric_scripts/):

```bash
# PPO
bash rubric_scripts/rurbichub_v1_Medical_ppo.sh

# GRPO
bash rubric_scripts/rurbichub_v1_Medical_grpo.sh

# GDPO
bash rubric_scripts/rurbichub_v1_Medical_gdpo.sh
```

More algorithm variants and scripts will be added over time.

## License 📄

Apache-2.0. See [`LICENSE`](LICENSE).

---

📫 Contact: zchen02@zju.edu.cn

<details>
<summary>Appendix 📎: sample rows from parquet datasets</summary>

The following examples are taken from the local parquet files in this repo:

- `data/health_bench/healthbench_val.parquet`
- `data/RubricHub_v1/RuRL/RuRL/rurbichub_v1_Medical.parquet`

To keep the README readable, long strings are truncated and rubric lists are summarized (`rubrics_len` + `rubrics_head`).

### healthbench_val.parquet (1 row)

```json
{
  "data_source": "healthbench",
  "ability": "medical_chat",
  "prompt": [
    {
      "role": "user",
      "content": "My 10-year-old has been having nighttime leg cramps this week. The pediatrician’s note says her potassium is 3.9 mmol/L, which is normal, and she might have a muscle strain from playing soccer daily. I want to write a MyChart message to her father explaining whether the cramps are from low potassium or strain."
    },
    {
      "role": "assistant",
      "content": "It looks like the pediatrician indicated that your child’s potassium level is within normal limits, so low potassium might not be the primary cause. Do you have any other lab results or details from the doctor’s note about her activity level or recommended treatments?"
    }
  ],
  "reward_model": {
    "style": "rubric",
    "ground_truth": "",
    "rubrics_len": 8,
    "rubrics_head": [
      {
        "criterion": "Correctly notes that calcium and potassium levels are normal, no supplements are needed, and muscle strain is the likely etiology.",
        "points": 8.0,
        "tags": {
          "axis": "accuracy",
          "cluster": null,
          "level": "example"
        }
      },
      {
        "criterion": "Fails to mention that calcium is normal and likely not causing leg cramps.",
        "points": -5.0,
        "tags": {
          "axis": "completeness",
          "cluster": null,
          "level": "example"
        }
      }
    ]
  }
}
```

### rurbichub_v1_Medical.parquet (1 row)

```json
{
  "data_source": "Medical",
  "ability": "Medical",
  "prompt": [
    {
      "role": "user",
      "content": "In which year was the WHO constitution made?"
    }
  ],
  "reward_model": {
    "style": "rubric",
    "ground_truth": "",
    "rubrics_len": 15,
    "rubrics_head": [
      {
        "criterion": "The response directly answers the question by providing the requested year without requiring inference.",
        "points": 9,
        "tags": {
          "function": "",
          "parameters": "",
          "verifier": "llm"
        }
      },
      {
        "criterion": "The response states that the WHO Constitution was adopted in 1946.",
        "points": 10,
        "tags": {
          "function": "",
          "parameters": "",
          "verifier": "llm"
        }
      }
    ]
  }
}
```

</details>
