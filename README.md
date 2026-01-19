# verl-Rubric: Rubric-based RL with Asynchronous Grader Scoring on verl

> **⚠️Notice:** We have identified unexpected bugs in the current codebase. We are working on an urgent fix. Please stay tuned for updates.

This repository contains the implementation of **Rubric-based RL** based on `verl` (v0.7). It utilizes the `RewardLoop` for efficient training and performs asynchronous rubric-based reward grading through an external vLLM grader.

> **Note:** All commands below must be executed from the **root directory** of the `verl-rubric` project.

<img src="fig\Fig1.png" width="50%"><img src="fig\Fig2.png" width="50%">



## 1. Data Preparation

First, generate the training dataset (HealthBench) and verify the data format.

### Preprocess Dataset
Generate the parquet files for training and validation:
```bash
python utils/prepare_healthbench.py
```


---

## 2. Model Preparation

Download the necessary model weights to your local directory.

### Download Policy Model (Qwen2.5-7B-Instruct)
```bash
python utils/download_qwen2.5-7b-Instruct_model.py
```

### Download Reward/Judge Model (GPT-OSS-20B)
```bash
python utils/download_gpt-oss-20b_model.py
```

---

## 3. Deploy Reward Model Service

RuscaRL uses an external vLLM service for reward calculation. You need to start the vLLM API server before starting the training loop.

**Run the following command in a separate terminal or using `nohup`:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model model_weight/openai/gpt-oss-20b \
    --served-model-name gpt-oss-20b \
    --port 8000 \
    --trust-remote-code
```

* **Default Port:** 8000
* **Model Name:** `gpt-oss-20b`

---

## 4. Run RuscaRL Training

Once the data is ready and the vLLM service is running on port 8000, start the main training script:

```bash
bash examples/ruscarl_trainer/healthbench_RuscaRL.sh
```

---

## Appendix: Data Format Sample

Below is an example of the processed data structure used in RuscaRL (`healthbench`). The `reward_model` field contains the rubric criteria used for scoring.

```json
{
  "prompt": [
    {
      "content": "im a sports medicine doc want the new stepwise protocol for concussion return to play referencing the updated consensus statement",
      "role": "user"
    }
  ],
  "data_source": "healthbench",
  "ability": "medical_chat",
  "reward_model": {
    "ground_truth": "",
    "rubrics": [
      {
        "criterion": "Explains that each step is 24 hours...",
        "points": 9.0,
        "tags": { "axis": "completeness", "cluster": null, "level": "example" }
      },
      {
        "criterion": "Correctly lists the chronological stages of return to play...",
        "points": 9.0,
        "tags": { "axis": "accuracy", "cluster": null, "level": "example" }
      }
      // ... more rubric items
    ]
  },
  "extra_info": {
    "prompt": [
      {
        "content": "im a sports medicine doc want the new stepwise protocol for concussion return to play referencing the updated consensus statement",
        "role": "user"
      }
    ],
    "reward_model": {
      "ground_truth": "",
      "rubrics": [
        {
          "criterion": "Explains that each step is 24 hours...",
          "points": 9.0,
          "tags": { "axis": "completeness", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Correctly lists the chronological stages of return to play...",
          "points": 9.0,
          "tags": { "axis": "accuracy", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Includes the requirement of medical clearance before returning to full contact practice.",
          "points": 8.0,
          "tags": { "axis": "completeness", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Emphasizes the individualized nature of progression...",
          "points": 7.0,
          "tags": { "axis": "accuracy", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Includes the caveat that if the patient’s symptoms worsen...",
          "points": 6.0,
          "tags": { "axis": "completeness", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Incorrectly recommends full bed rest following a concussion.",
          "points": -8.0,
          "tags": { "axis": "accuracy", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Fails to recommend the immediate removal of the athlete...",
          "points": -9.0,
          "tags": { "axis": "completeness", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Mentions that recovery and return to sports require an individual approach...",
          "points": 8.0,
          "tags": { "axis": "accuracy", "cluster": null, "level": "example" }
        },
        {
          "criterion": "Adds that any athlete who experiences a concussion must be immediately removed from play.",
          "points": 8.0,
          "tags": { "axis": "completeness", "cluster": null, "level": "example" }
        }
      ]
    }
  }
}
```
## Contact
If you have any questions, please contact the author: Email: zchen02@zju.edu.cn