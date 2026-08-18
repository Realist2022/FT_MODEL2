# src/evaluate.py

import json
from typing import Any, Dict, cast

from config import paths, training
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class JsonEvaluator:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(model_path, device_map="auto"),
        )

    def generate(self, instruction: str, max_new_tokens: int = 512) -> str:
        prompt = (
            "You are a strict JSON generator.\n"
            "Given the instruction, output ONLY the JSON object.\n\n"
            f"Instruction:\n{instruction}\n\n"
            "JSON output:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip prompt to get only model completion
        return text[len(prompt):].strip()

    @staticmethod
    def is_valid_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def check_schema(obj: Dict) -> bool:
        if "skills_evaluation" not in obj:
            return False
        se = obj["skills_evaluation"]
        required_keys = [
            "requirement_category",
            "job_requirements",
            "matched_cv_skills",
            "missing_cv_skills",
            "rationale",
        ]
        return all(k in se for k in required_keys)

    def evaluate_dataset(self, data_file: str, num_samples: int = 100) -> None:
        ds = load_dataset("json", data_files={"test": data_file})["test"]
        ds = ds.select(range(min(num_samples, len(ds))))

        valid_json_count = 0
        schema_ok_count = 0

        for i, ex in enumerate(ds):
            example = cast(Dict[str, Any], ex)
            instruction = str(example["instruction"])
            completion = self.generate(instruction)

            if not self.is_valid_json(completion):
                print(f"[{i}] Invalid JSON:\n{completion}\n")
                continue

            obj = json.loads(completion)
            valid_json_count += 1

            if self.check_schema(obj):
                schema_ok_count += 1
            else:
                print(f"[{i}] JSON valid but schema mismatch:\n{completion}\n")

        print(f"Total samples: {len(ds)}")
        print(f"Valid JSON: {valid_json_count}")
        print(f"Schema OK: {schema_ok_count}")


def main():
    evaluator = JsonEvaluator(model_path=training.output_dir)
    evaluator.evaluate_dataset(data_file=paths.test_file, num_samples=50)


if __name__ == "__main__":
    main()
