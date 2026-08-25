# src/evaluate.py

import json
from pathlib import Path
from typing import Any, Dict, List, cast
import matplotlib.pyplot as plt
from datasets import load_dataset

from src.core.config import paths, training
from src.core.schema import (
    JobRequirementsOutput,
    OverallExperienceResponse,
    SkillEvaluationDecision,
)
from src.pipeline.inference import ModelRunner, ResponseParser

TASK_SCHEMAS = {
    "skill_match": SkillEvaluationDecision,
    "requirements": JobRequirementsOutput,
    "experience": OverallExperienceResponse,
}


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


class JsonEvaluator:
    """Benchmarks model generations per agent task (skill_match /
    requirements / experience) on JSON validity, schema compliance, and
    task-specific quality metrics."""

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

        stats: Dict[str, Dict[str, Any]] = {}

        for i, ex in enumerate(ds):
            record = cast(Dict[str, Any], ex)
            task = record.get("task", "skill_match")
            acc = stats.setdefault(task, {
                "n": 0, "json_valid": 0, "schema_valid": 0,
                "accuracy": [], "precision": [], "recall": [],
                "title_match": [], "years_match": [], "relevance_accuracy": [],
            })
            acc["n"] += 1

            prompt_messages, ground_truth = ResponseParser.split_messages_and_ground_truth(record)

            if hasattr(self.runner, "generate_chat"):
                raw_output = self.runner.generate_chat(prompt_messages)
            else:
                # Runners without chat support (e.g. the Gemini baseline)
                # get the system and user content as one plain prompt.
                raw_output = self.runner.generate(
                    "\n\n".join(m["content"] for m in prompt_messages)
                )
            parsed_json = ResponseParser.extract_json(raw_output)

            print(f"[{i + 1}/{num_eval}] Evaluating {task} sample...")

            if parsed_json is None:
                print(f" -> Invalid JSON output:\n{raw_output[:150]}...")
                continue

            acc["json_valid"] += 1

            try:
                TASK_SCHEMAS[task].model_validate(parsed_json)
                schema_ok = True
            except Exception:
                schema_ok = False

            if task == "skill_match":
                self._score_skill_match(ground_truth, parsed_json, schema_ok, acc)
            elif task == "requirements":
                if schema_ok:
                    acc["schema_valid"] += 1
                self._score_requirements(ground_truth, parsed_json, acc)
            elif task == "experience":
                if schema_ok:
                    acc["schema_valid"] += 1
                self._score_experience(ground_truth, parsed_json, acc)

        results: Dict[str, Any] = {"total_samples": num_eval, "per_task": {}}
        for task, acc in stats.items():
            precision = _mean(acc["precision"])
            recall = _mean(acc["recall"])
            task_metrics = {
                "samples": acc["n"],
                "json_validity_rate": _safe_rate(acc["json_valid"], acc["n"]),
                "schema_compliance_rate": _safe_rate(acc["schema_valid"], acc["n"]),
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
            if task == "skill_match":
                task_metrics["decision_accuracy"] = _mean(acc["accuracy"])
            if task == "experience":
                task_metrics["title_match_rate"] = _mean(acc["title_match"])
                task_metrics["years_match_rate"] = _mean(acc["years_match"])
                task_metrics["relevance_accuracy"] = _mean(acc["relevance_accuracy"])
            results["per_task"][task] = task_metrics

        self._print_summary(results)
        self._plot_metrics(results)
        return results

    # ------------------------------------------------------------------
    # Per-task scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_skill_match(
        ground_truth: Dict[str, Any], parsed_json: Dict[str, Any], schema_ok: bool, acc: Dict
    ) -> None:
        gt_map = {
            e["requirement_id"]: bool(e["matched"])
            for e in ground_truth.get("evaluations", [])
        }
        pred_evals = parsed_json.get("evaluations", [])
        pred_ids = [e.get("requirement_id") for e in pred_evals if isinstance(e, dict)]

        # Schema compliance includes exact requirement_id coverage
        # (one entry per supplied id, no duplicates, no extras).
        covers_ids = len(pred_ids) == len(set(pred_ids)) and set(pred_ids) == set(gt_map)
        if schema_ok and covers_ids:
            acc["schema_valid"] += 1

        pred_map = {
            e["requirement_id"]: bool(e["matched"])
            for e in pred_evals
            if isinstance(e, dict) and "requirement_id" in e and "matched" in e
        }

        if gt_map:
            correct = sum(1 for rid, m in gt_map.items() if pred_map.get(rid) == m)
            acc["accuracy"].append(correct / len(gt_map))

        gt_matched = {rid for rid, m in gt_map.items() if m}
        pred_matched = {rid for rid, m in pred_map.items() if m}
        if gt_matched or pred_matched:
            hit = len(gt_matched & pred_matched)
            acc["precision"].append(_safe_rate(hit, len(pred_matched)))
            acc["recall"].append(_safe_rate(hit, len(gt_matched)))

    @staticmethod
    def _score_requirements(
        ground_truth: Dict[str, Any], parsed_json: Dict[str, Any], acc: Dict
    ) -> None:
        def names(obj: Dict[str, Any]) -> set:
            return {
                str(r.get("skill_name", "")).strip().casefold()
                for r in obj.get("job_requirements", [])
                if isinstance(r, dict)
            } - {""}

        gt_names = names(ground_truth)
        pred_names = names(parsed_json)
        if gt_names or pred_names:
            hit = len(gt_names & pred_names)
            acc["precision"].append(_safe_rate(hit, len(pred_names)))
            acc["recall"].append(_safe_rate(hit, len(gt_names)))

    @staticmethod
    def _score_experience(
        ground_truth: Dict[str, Any], parsed_json: Dict[str, Any], acc: Dict
    ) -> None:
        gt = ground_truth.get("overall_experience", {})
        pred = parsed_json.get("overall_experience", {})
        if not isinstance(pred, dict):
            pred = {}

        acc["title_match"].append(
            1.0 if pred.get("target_job_title") == gt.get("target_job_title") else 0.0
        )
        acc["years_match"].append(
            1.0 if pred.get("target_overall_years") == gt.get("target_overall_years") else 0.0
        )

        def role_key(role: Dict[str, Any]):
            return (
                str(role.get("role_title", "")).strip().casefold(),
                role.get("start_date"),
                role.get("end_date"),
            )

        gt_roles = {role_key(r): r for r in gt.get("candidate_roles", [])}
        pred_roles = {
            role_key(r): r for r in pred.get("candidate_roles", []) if isinstance(r, dict)
        }

        if gt_roles or pred_roles:
            hits = set(gt_roles) & set(pred_roles)
            acc["precision"].append(_safe_rate(len(hits), len(pred_roles)))
            acc["recall"].append(_safe_rate(len(hits), len(gt_roles)))
            if hits:
                agree = sum(
                    1 for key in hits
                    if bool(pred_roles[key].get("is_relevant")) == bool(gt_roles[key].get("is_relevant"))
                )
                acc["relevance_accuracy"].append(agree / len(hits))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(metrics: Dict[str, Any]) -> None:
        print("\n========================================")
        print("          EVALUATION RESULTS            ")
        print("========================================")
        print(f"Total Evaluated Samples : {metrics['total_samples']}")
        for task, m in metrics["per_task"].items():
            print(f"\n--- {task} ({m['samples']} samples) ---")
            print(f"JSON Syntax Validity    : {m['json_validity_rate'] * 100:.2f}%")
            print(f"Pydantic Schema Match   : {m['schema_compliance_rate'] * 100:.2f}%")
            if "decision_accuracy" in m:
                print(f"Decision Accuracy       : {m['decision_accuracy'] * 100:.2f}%")
            print(f"Precision               : {m['precision'] * 100:.2f}%")
            print(f"Recall                  : {m['recall'] * 100:.2f}%")
            print(f"F1 Score                : {m['f1'] * 100:.2f}%")
            if "title_match_rate" in m:
                print(f"Target Title Match      : {m['title_match_rate'] * 100:.2f}%")
                print(f"Target Years Match      : {m['years_match_rate'] * 100:.2f}%")
                print(f"Relevance Accuracy      : {m['relevance_accuracy'] * 100:.2f}%")
        print("========================================\n")

    @staticmethod
    def _plot_metrics(
        metrics: Dict[str, Any], output_path: str = "artifacts/evaluation_metrics.png"
    ) -> None:
        """Generates and saves a grouped bar chart of per-task metrics."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        bar_labels: List[str] = []
        scores: List[float] = []
        colors: List[str] = []
        task_colors = {"skill_match": "#4A90E2", "requirements": "#F5A623", "experience": "#7ED321"}

        for task, m in metrics["per_task"].items():
            shown = [
                ("JSON", m["json_validity_rate"]),
                ("Schema", m["schema_compliance_rate"]),
                ("F1", m["f1"]),
            ]
            if "decision_accuracy" in m:
                shown.append(("Accuracy", m["decision_accuracy"]))
            if "relevance_accuracy" in m:
                shown.append(("Relevance", m["relevance_accuracy"]))
            for name, value in shown:
                bar_labels.append(f"{task}\n{name}")
                scores.append(value * 100)
                colors.append(task_colors.get(task, "#50E3C2"))

        plt.figure(figsize=(max(9, 1.1 * len(bar_labels)), 5))
        bars = plt.bar(bar_labels, scores, color=colors, width=0.55)
        plt.ylim(0, 110)
        plt.ylabel("Score (%)")
        plt.title(
            f"FT_MODEL2 Evaluation Summary (N={metrics['total_samples']})",
            fontsize=13,
            fontweight="bold",
        )
        plt.xticks(fontsize=8)

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
                fontsize=8,
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
