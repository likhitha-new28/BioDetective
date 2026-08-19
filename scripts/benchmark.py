"""Run the deterministic BioDetective synthetic benchmark."""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biodetective.synthetic.benchmark import run_synthetic_benchmark


if __name__ == "__main__":
    print(json.dumps(run_synthetic_benchmark(), indent=2, sort_keys=True))
