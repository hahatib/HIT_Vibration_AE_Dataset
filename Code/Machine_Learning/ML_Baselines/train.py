from pathlib import Path
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from common.comparison.workflow import ml_main


def main(argv=None):
    ml_main(ROOT, argv)


if __name__ == "__main__":
    main()
