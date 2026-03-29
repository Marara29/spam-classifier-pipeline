from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

REQUIRED_COLUMNS = ["Message ID", "Subject", "Message", "Spam/Ham", "Date"]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the raw Enron dataset."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Message ID"] = df["Message ID"].astype(int)
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["Spam/Ham"] = df["Spam/Ham"].str.lower().str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_jsonl(records: Iterable[Mapping], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
