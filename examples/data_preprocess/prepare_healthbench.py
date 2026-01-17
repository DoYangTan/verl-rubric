import os
import json
import datasets
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: Dict[str, Any]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        tags_data = d.get("tags", [])
        if isinstance(tags_data, list):
            tags_dict = {}
            for tag in tags_data:
                if isinstance(tag, str) and ":" in tag:
                    key, value = tag.split(":", 1)
                    tags_dict[key] = value
                elif isinstance(tag, str):
                    tags_dict[tag] = True
            tags_data = tags_dict
        elif not isinstance(tags_data, dict):
            tags_data = {}
            
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=tags_data
        )

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags
        }

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        prompt = example['prompt']
        
        rubrics = [RubricItem.from_dict(r) for r in example['rubrics']]
        
        reward_model = {
            "style": "rubric",
            "rubrics": [r.to_dict() for r in rubrics],
            "ground_truth": ""
        }
        
        data = {
            "data_source": "healthbench",
            "prompt": prompt,
            "ability": "medical_chat",
            "reward_model": reward_model
        }
        return data
    
    return process_fn

def process_dataset(data_list: List[Dict[str, Any]], split: str) -> datasets.Dataset:
    dataset = datasets.Dataset.from_list(data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names
    )
    
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    return shuffled_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='ruscarl/raw_data/healthbench')
    parser.add_argument('--output_dir', default='data/health_bench')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    train_file = os.path.join(args.local_dir, 'healthbench_train.jsonl')
    train_data = load_jsonl(train_file)
    
    val_file = os.path.join(args.local_dir, 'healthbench_eval.jsonl')
    val_data = load_jsonl(val_file)
    
    train_dataset = process_dataset(train_data, 'train')
    val_dataset = process_dataset(val_data, 'val')
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_train.parquet'))
    val_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_val.parquet'))
    
    print("\nDataset Information:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    print("\nTraining set sample example:")
    print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))
    print("\nValidation set sample example:")
    print(json.dumps(val_dataset[0], indent=2, ensure_ascii=False))
    
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main()