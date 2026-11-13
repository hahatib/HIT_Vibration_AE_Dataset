"""Run directly in PyCharm or from the command line."""

from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from common.runtime import load_config
from common.primary.workflow import run_experiment, dry_run
from DC_Offset_Control.DC_Removed import adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("experiment.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, output=args.output, device=args.device)
    if args.dry_run:
        print(json.dumps(dry_run(config, adapter), indent=2, ensure_ascii=False))
    else:
        run_experiment(config, adapter)


if __name__ == "__main__":
    main()
