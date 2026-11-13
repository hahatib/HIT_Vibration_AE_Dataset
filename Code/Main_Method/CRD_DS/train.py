"""Run in PyCharm, or: python train.py --config experiment.json --device cuda."""

from pathlib import Path
import argparse
import json
import os
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from common.runtime import load_config
from common.primary.workflow import run_experiment, dry_run
from Main_Method.CRD_DS.experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("experiment.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, output=args.output, device=args.device)
    experiment = Experiment()
    if args.dry_run:
        print(json.dumps(dry_run(config, experiment), indent=2, ensure_ascii=False))
    else:
        run_experiment(config, experiment)


if __name__ == "__main__":
    main()
