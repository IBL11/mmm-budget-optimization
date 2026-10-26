"""Shared helpers: paths and config loading."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
CONFIG = ROOT / "config" / "params.json"
RAW_FILE = DATA_RAW / "mmm_dataset.xlsx"


def load_params() -> dict:
    with open(CONFIG) as f:
        return json.load(f)


def save_params(params: dict) -> None:
    with open(CONFIG, "w") as f:
        json.dump(params, f, indent=2)


def load_raw():
    """Load the raw MMM dataset with parsed dates."""
    import pandas as pd
    df = pd.read_excel(RAW_FILE, sheet_name="MMM_Data", parse_dates=["month"])
    return df.sort_values("month").reset_index(drop=True)
