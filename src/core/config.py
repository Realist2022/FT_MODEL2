# src/config.py

from dataclasses import dataclass

@dataclass
class PathsConfig:
    raw_esco_skills: str = "data/raw/esco_skills.json"
    raw_esco_occupations: str = "data/raw/esco_occupations.json"
    processed_dataset: str = "data/processed/dataset.jsonl"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    test_file: str = "data/processed/test.jsonl"

@dataclass
class TrainingConfig:
    base_model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    output_dir: str = "models/checkpoints/llama3.2-3b-esco-json"
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_seq_length: int = 2048
    use_4bit_qlora: bool = True

paths = PathsConfig()
training = TrainingConfig()
