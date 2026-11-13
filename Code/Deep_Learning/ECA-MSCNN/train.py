from pathlib import Path
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from common.comparison.workflow import deep_main


def main(argv=None):
    deep_main(ROOT, "ECA-MSCNN", argv)


if __name__ == "__main__":
    main()
