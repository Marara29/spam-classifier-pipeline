from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enron_spam_pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Enron spam classification pipeline.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the raw CSV file containing the Enron spam dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where TSV artifacts and metrics will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(data_path=args.data, output_dir=args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
