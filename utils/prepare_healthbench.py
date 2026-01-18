import os
import json
import argparse
import datasets
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: Dict[str, str]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        tags_data = d.get("tags", {})
        final_tags = {}

        if isinstance(tags_data, list):
            for tag in tags_data:
                if isinstance(tag, str):
                    if ":" in tag:
                        key, value = tag.split(":", 1)
                        final_tags[key] = str(value)
                    else:
                        final_tags[tag] = "True"
        elif isinstance(tags_data, dict):
            for k, v in tags_data.items():
                final_tags[str(k)] = str(v)

        return cls(
            criterion=str(d.get("criterion", "")),
            points=float(d.get("points", 0.0)),
            tags=final_tags
        )

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags
        }

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data

def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        prompt = example.get('prompt', "")
        
        raw_rubrics = example.get('rubrics', [])
        if raw_rubrics is None:
            raw_rubrics = []
            
        rubrics = [RubricItem.from_dict(r) for r in raw_rubrics]
        rubrics_dicts = [r.to_dict() for r in rubrics]

        reward_model = {
            "style": "rubric",
            "rubrics": rubrics_dicts,
            "ground_truth": example.get("ground_truth", "")
        }
        
        extra_info = {
            "prompt": prompt,
            "reward_model": reward_model
        }

        return {
            "data_source": "healthbench",
            "prompt": prompt,
            "ability": "medical_chat",
            "reward_model": reward_model,
            "extra_info": extra_info 
        }
    
    return process_fn

def process_dataset(data_list: List[Dict[str, Any]], split: str) -> datasets.Dataset:
    if not data_list:
        print(f"Warning: Data list for {split} is empty!")
        return datasets.Dataset.from_list([])

    dataset = datasets.Dataset.from_list(data_list)
    
    print(f"Mapping {split} dataset...")
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names, 
        load_from_cache_file=False
    )
    
    return processed_dataset.shuffle(seed=42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='raw_data/healthbench')
    parser.add_argument('--output_dir', default='data/health_bench')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_file = os.path.join(args.local_dir, 'healthbench_train.jsonl')
    val_file = os.path.join(args.local_dir, 'healthbench_eval.jsonl')
    
    print(f"Loading files from {args.local_dir}...")
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    
    train_dataset = process_dataset(train_data, 'train')
    val_dataset = process_dataset(val_data, 'val')
    
    train_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_train.parquet'))
    val_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_val.parquet'))
    
    print(f"Successfully saved parquet files to {args.output_dir}")

    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        try:
            makedirs(args.hdfs_dir)
            copy(src=args.output_dir, dst=args.hdfs_dir)
            print(f"Copied to HDFS: {args.hdfs_dir}")
        except Exception as e:
            print(f"HDFS Copy failed: {e}")

if __name__ == '__main__':
    main()