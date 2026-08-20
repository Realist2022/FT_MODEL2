# main.py

import argparse
from importlib import import_module

STAGES = {
    "01": ("src.extraction", "Step 01: Extraction"),
    "02": ("src.synthesize", "Step 02: Synthesis"),
    "03": ("src.split", "Step 03: Dataset Split"),
    "04": ("src.train", "Step 04: LoRA Training"),
    "05": ("src.pipeline.evaluate", "Step 05: Model Evaluation"),
    "06": ("src.pipeline.evaluate_gemini", "Step 06: Gemini Model Evaluation"),
    "07": ("src.pipeline.export", "Step 07: Export LoRA Adapter"),
}


def run_stage(stage: str) -> None:
    module_name, label = STAGES[stage]
    print(f"\n{'=' * 40}")
    print(f" Running {label}")
    print(f"{'=' * 40}\n")
    module = import_module(module_name)
    if hasattr(module, "run"):
        module.run()
    elif hasattr(module, "upload_lora"):
        module.upload_lora()
    else:
        raise AttributeError(f"Module '{module_name}' does not define a 'run()' or 'upload_lora()' function.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FT_MODEL2 training and evaluation pipeline.")
    parser.add_argument(
        "stage",
        nargs="?",
        choices=[*STAGES.keys(), "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    stages_to_run = list(STAGES.keys()) if args.stage == "all" else [args.stage]
    for stage in stages_to_run:
        run_stage(stage)


if __name__ == "__main__":
    main()