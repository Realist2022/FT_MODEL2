# src/evaluate.py

import json
from pathlib import Path
from typing import Any, Dict, List, cast
import matplotlib.pyplot as plt
from datasets import load_dataset

from src.core.config import paths, training
from src.utils.parser import JsonParser
from src.pipeline.inference import ModelRunner, ResponseParser


class JsonEvaluator:
    """Evaluator that benchmarks model generations on accuracy, schema validity, and skill matching."""

    def __init__(self, model_path: str | None = None):
        checkpoint = model_path or training.output_dir
        self.runner = ModelRunner(
            model_path=checkpoint,
            max_seq_length=training.max_seq_length,
            max_new_tokens=1024,
        )

    def evaluate_dataset(self, data_file: str | None = None, num_samples: int = 50) -> Dict[str, Any]:
        target_file = data_file or paths.test_file
        print(f"\n--- Loading evaluation dataset from {target_file} ---")

        ds = load_dataset("json", data_files={"test": target_file})["test"]
        num_eval = min(num_samples, len(ds))  # type: ignore
        ds = ds.select(range(num_eval))       # type: ignore

        valid_json_count = 0
        schema_valid_count = 0

        matched_precision_scores: List[float] = []
        matched_recall_scores: List[float] = []

        for i, ex in enumerate(ds):
            record = cast(Dict[str, Any], ex)
            prompt, ground_truth = ResponseParser.split_prompt_and_ground_truth(record)

            raw_output = self.runner.generate(prompt)
            parsed_json = ResponseParser.extract_json(raw_output)

            print(f"[{i + 1}/{num_eval}] Evaluating sample...")

            if parsed_json is None:
                print(f" -> Invalid JSON output:\n{raw_output[:150]}...")
                continue

            valid_json_count += 1

            # Validate against Pydantic schema
            raw_json_str = json.dumps(parsed_json)
            if JsonParser.validate_json(raw_json_str):
                schema_valid_count += 1

            # Compare skill prediction accuracy
            gt_eval = ground_truth.get("skills_evaluation", {})
            pred_eval = parsed_json.get("skills_evaluation", {})

            gt_matched = set(gt_eval.get("matched_cv_skills", []))
            pred_matched = set(pred_eval.get("matched_cv_skills", []))

            if pred_matched or gt_matched:
                intersection = gt_matched.intersection(pred_matched)
                precision = len(intersection) / len(pred_matched) if pred_matched else 0.0
                recall = len(intersection) / len(gt_matched) if gt_matched else 0.0

                matched_precision_scores.append(precision)
                matched_recall_scores.append(recall)

        avg_precision = (
            sum(matched_precision_scores) / len(matched_precision_scores)
            if matched_precision_scores
            else 0.0
        )
        avg_recall = (
            sum(matched_recall_scores) / len(matched_recall_scores)
            if matched_recall_scores
            else 0.0
        )
        avg_f1 = (
            (2 * avg_precision * avg_recall / (avg_precision + avg_recall))
            if (avg_precision + avg_recall) > 0
            else 0.0
        )

        results = {
            "total_samples": num_eval,
            "json_validity_rate": valid_json_count / num_eval if num_eval > 0 else 0.0,
            "schema_compliance_rate": schema_valid_count / num_eval if num_eval > 0 else 0.0,
            "matched_skills_precision": avg_precision,
            "matched_skills_recall": avg_recall,
            "matched_skills_f1": avg_f1,
        }

        self._print_summary(results)
        self._plot_metrics(results)
        return results

    @staticmethod
    def _print_summary(metrics: Dict[str, Any]) -> None:
        print("\n========================================")
        print("          EVALUATION RESULTS            ")
        print("========================================")
        print(f"Total Evaluated Samples : {metrics['total_samples']}")
        print(f"JSON Syntax Validity    : {metrics['json_validity_rate'] * 100:.2f}%")
        print(f"Pydantic Schema Match   : {metrics['schema_compliance_rate'] * 100:.2f}%")
        print(f"Skill Precision         : {metrics['matched_skills_precision'] * 100:.2f}%")
        print(f"Skill Recall            : {metrics['matched_skills_recall'] * 100:.2f}%")
        print(f"Skill F1 Score          : {metrics['matched_skills_f1'] * 100:.2f}%")
        print("========================================\n")

    @staticmethod
    def _plot_metrics(
        metrics: Dict[str, Any], output_path: str = "artifacts/evaluation_metrics.png"
    ) -> None:
        """Generates and saves a bar chart of the evaluation metrics."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        labels = ["JSON Validity", "Schema Match", "Precision", "Recall", "F1 Score"]
        scores = [
            metrics["json_validity_rate"] * 100,
            metrics["schema_compliance_rate"] * 100,
            metrics["matched_skills_precision"] * 100,
            metrics["matched_skills_recall"] * 100,
            metrics["matched_skills_f1"] * 100,
        ]

        colors = ["#4A90E2", "#50E3C2", "#F5A623", "#F8E71C", "#7ED321"]

        plt.figure(figsize=(9, 5))
        bars = plt.bar(labels, scores, color=colors, width=0.55)
        plt.ylim(0, 110)
        plt.ylabel("Score (%)")
        plt.title(
            f"FT_MODEL2 Evaluation Summary (N={metrics['total_samples']})",
            fontsize=13,
            fontweight="bold",
        )

        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.show()
        plt.close()
        print(f"📈 Chart successfully saved to: {output_path}")


def run() -> None:
    """Stage runner function called by main.py orchestrator."""
    evaluator = JsonEvaluator()
    evaluator.evaluate_dataset(num_samples=50)


if __name__ == "__main__":
    run()