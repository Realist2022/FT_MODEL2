# src/split.py

import json
import random
from typing import List
from config import paths


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

        self._write_jsonl(paths.train_file, train)
        self._write_jsonl(paths.val_file, val)
        self._write_jsonl(paths.test_file, test)

        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    @staticmethod
    def _write_jsonl(path: str, items: List[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    splitter = DatasetSplitter(input_file=paths.processed_dataset)
    splitter.split()


if __name__ == "__main__":
    main()
