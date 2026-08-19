# src/split.py

import os
import json
import random
from typing import List

from dotenv import load_dotenv
from datasets import Dataset, DatasetDict

from src.core.config import paths


class DatasetSplitter:
    def __init__(self, input_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
        self.input_file = input_file
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def _load_lines(self) -> List[dict]:
        items = []
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def split(self) -> None:
        items = self._load_lines()
        random.shuffle(items)

        n = len(items)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_test = n - n_train - n_val

        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]

        # 1. Save locally (Keeping your original functionality)
        self._write_jsonl(paths.train_file, train)
        self._write_jsonl(paths.val_file, val)
        self._write_jsonl(paths.test_file, test)
        print(f"Local files saved. Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        # 2. Push to Hugging Face Hub
        self._push_to_hub(train, val, test)

    def _push_to_hub(self, train: List[dict], val: List[dict], test: List[dict]) -> None:
        hf_token = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_DATASET_REPO", "Realist2026/cv-jd-semantic-splits") # Adjust default as needed

        if not hf_token:
            print("No HF_TOKEN found in environment. Skipping Hugging Face upload.")
            return

        print("Converting to Hugging Face datasets...")
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test)
        })

        print(f"Pushing dataset to Hugging Face Hub at {repo_id}...")
        dataset_dict.push_to_hub(repo_id, token=hf_token)
        print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")


    @staticmethod
    def _write_jsonl(path: str, items: List[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run():
    load_dotenv() # Load environment variables for HF_TOKEN
    splitter = DatasetSplitter(input_file=paths.processed_dataset)
    splitter.split()

if __name__ == "__main__":
    run()