"""Choose an experiment below and run this file in an IDE or from a terminal."""

import argparse
import os
from pathlib import Path
import runpy
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "crd_ds": "Main_Method/CRD_DS",
    "cwru": "CWRU_Validation",
    "dc_removed": "DC_Offset_Control/DC_Removed",
    "dc_retained": "DC_Offset_Control/DC_Retained",
    "ml": "Machine_Learning/ML_Baselines",
    "de": "Deep_Learning/DE-1D-CNN",
    "attention": "Deep_Learning/Attention-1DCNN",
    "eca": "Deep_Learning/ECA-MSCNN",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=EXPERIMENT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    args = parser.parse_args()
    entry = ROOT / EXPERIMENTS[args.experiment] / "train.py"
    arguments = [str(entry)]
    for key in ("config", "output"):
        if getattr(args, key) is not None:
            arguments += [
                f"--{key}",
                str(Path(getattr(args, key)).expanduser().resolve()),
            ]
    if args.device != "auto":
        arguments += ["--device", args.device]
    if args.dry_run:
        arguments.append("--dry-run")
    original_arguments = sys.argv
    try:
        sys.argv = arguments
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        sys.argv = original_arguments


if __name__ == "__main__":
    # PyCharm: edit these values and run this file. No terminal arguments are required.
    EXPERIMENT = "crd_ds"
    CONFIG = None
    OUTPUT = None
    DEVICE = "auto"
    DRY_RUN = False
    main()
