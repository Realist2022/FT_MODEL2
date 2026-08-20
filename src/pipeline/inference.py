# src/inference.py

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from unsloth import FastLanguageModel


class ModelRunner:
    """Handles loading the fine-tuned LoRA model and generating completions."""

    def __init__(
        self,
        model_path: Path | str,
        max_seq_length: int = 2048,
        max_new_tokens: int = 1024,
    ):
        self.model_path = str(model_path)
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self._load()

    def _load(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]
        available = self.max_seq_length - prompt_len

        if available <= 0:
            raise ValueError(
                f"Prompt token length ({prompt_len}) exceeds max sequence length ({self.max_seq_length})."
            )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(self.max_new_tokens, available),
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)


class ResponseParser:
    """Extracts JSON from model output and formats input records into exact training prompts."""

    @staticmethod
    def extract_json(raw: str) -> Dict[str, Any] | None:
        """Finds and parses the first complete JSON object in the model's response."""
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if 0 <= start < end:
                candidate = raw[start : end + 1]
                return json.loads(candidate)
            return None
        except Exception:
            return None

    @staticmethod
    def split_prompt_and_ground_truth(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Reconstructs the prompt format used during training and separates the ground truth."""
        instruction = record.get("instruction", "")
        ground_truth = record.get("output", {})

        prompt = (
            "You are a strict JSON generator.\n"
            "Given the instruction, output ONLY the JSON object.\n\n"
            f"Instruction:\n{instruction}\n\n"
            "JSON output:\n"
        )

        return prompt, ground_truth


def run() -> None:
    """Standalone run entry point for inference test."""
    from src.core.config import training

    print(f"Loading checkpoint from: {training.output_dir}")
    runner = ModelRunner(model_path=training.output_dir)

    sample_instruction = (
        "Role: Software Engineer\n"
        "Job Requirements: Python, FastAPI, Docker, SQL\n"
        "Candidate Skills: Python, SQL, Git"
    )
    prompt = (
        "You are a strict JSON generator.\n"
        "Given the instruction, output ONLY the JSON object.\n\n"
        f"Instruction:\n{sample_instruction}\n\n"
        "JSON output:\n"
    )

    print("\n--- Generating sample inference ---")
    raw_output = runner.generate(prompt)
    parsed = ResponseParser.extract_json(raw_output)

    print("\nRaw Output:\n", raw_output)
    print("\nParsed JSON:\n", json.dumps(parsed, indent=2) if parsed else "Failed to parse JSON")


if __name__ == "__main__":
    run()