# src/utils/utils.py

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_text(path: str | Path) -> str:
    """Read a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str | Path, content: str) -> None:
    """Write a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON safely."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Dict[str, Any], pretty: bool = False) -> None:
    """Save JSON with optional pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, f, ensure_ascii=False)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def timer(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper
