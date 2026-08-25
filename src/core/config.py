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
    paraphrase_cache: str = "data/processed/paraphrase_cache.json"

@dataclass
class SynthesisConfig:
    # Fraction of MATCHED skill mentions that get a genuinely differently-worded
    # (LLM-generated) real-world equivalent instead of the literal ESCO name/alt_label,
    # when an LLM paraphraser is available. Distractors/near-miss mentions never use
    # this -- they must stay literal/near-literal so the model still learns to reject
    # domain-plausible-but-wrong mentions. See SkillParaphraser in synthesize.py.
    llm_paraphrase_prob: float = 0.35
    # Fraction of matched skills (when not LLM-paraphrased) that use an ESCO alt_label
    # instead of the preferred name -- unchanged from the original behaviour.
    alt_label_prob: float = 0.4

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
synthesis = SynthesisConfig()
training = TrainingConfig()
