"""CS-456 CCP — single-command entry point for the whole pipeline.

Examples:
    python run_all.py                 # run all 5 tasks (real data, then build report)
    python run_all.py --task 1        # run only Task 1
    python run_all.py --data synthetic  # force fully-offline synthetic data
    python run_all.py --no-report     # skip the Task-5 report generation

Every task writes its visual outputs under outputs/task<N>/ and a machine-readable
summary to outputs/results.json.
"""

import argparse
import json

from src import config
from src.utils import ensure_dirs, log, set_seed


def _json_safe(obj):
    """Best-effort conversion of numpy / tuple structures to JSON-native types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="CS-456 CCP pipeline runner")
    parser.add_argument("--task", default="all", choices=["all", "1", "2", "3", "4"],
                        help="which task to run (default: all)")
    parser.add_argument("--data", default=None, choices=["auto", "synthetic", "real"],
                        help="data-sourcing mode (overrides config.DATA_MODE)")
    parser.add_argument("--no-report", action="store_true", help="skip Task-5 report")
    args = parser.parse_args()

    if args.data:
        config.DATA_MODE = args.data
    set_seed()
    ensure_dirs()
    log(f"CS-456 CCP pipeline | task={args.task} | data_mode={config.DATA_MODE}")

    todo = ["1", "2", "3", "4"] if args.task == "all" else [args.task]
    results = {}

    if "1" in todo:
        from src.task1_edges import run as run1
        results["task1"] = run1()
    if "2" in todo:
        from src.task2_sift import run as run2
        results["task2"] = run2()
    if "3" in todo:
        from src.task3_panorama import run as run3
        results["task3"] = run3()
    if "4" in todo:
        from src.task4_cnn import run as run4
        results["task4"] = run4()

    results_path = config.OUTPUTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    log(f"wrote consolidated metrics to {results_path}")

    if args.task == "all" and not args.no_report:
        from src.task5_report import generate
        generate(results)

    log("DONE.")


if __name__ == "__main__":
    main()
