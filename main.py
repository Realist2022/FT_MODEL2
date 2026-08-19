import argparse
from importlib import import_module


STAGES = {
    "01": ("src.pipeline.extraction", "Step 01: Extraction"),
    "02": ("src.pipeline.synthesize", "Step 02: Synthesis"),
    "03": ("src.pipeline.split", "Step 03: Split"),
    "04": ("src.pipeline.train", "Step 04: Training"),
    "05": ("src.pipeline.evaluate", "Step 05: Evaluation"),
    "06": ("src.pipeline.evaluate", "Step 06: Evaluation"),
    "07": ("src.pipeline.inference", "Step 07: Inference"),
}

def run_stage(stage: str) -> None:
    module_name, label = STAGES[stage]
    print(label)
    import_module(module_name).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ESCO model pipeline.")
    parser.add_argument(
        "stage",
        nargs="?",
        choices=[*STAGES, "all"],
        default="all",
        help="pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_stage(stage)


if __name__ == "__main__":
    main()
