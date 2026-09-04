from __future__ import annotations

from pathlib import Path

import pandas as pd

from matcher.models import ProductRecord
from matcher.normalize import normalize_product


INPUT_COLUMNS = [
    "item_id",
    "name",
    "brand_raw",
    "category",
    "department",
    "item_info",
    "subcategory",
    "sizing_comp",
]


def load_products(path: Path) -> list[ProductRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"item_id", "name"}
    missing = required.difference(header)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    selected = [column for column in INPUT_COLUMNS if column in header]

    frame = pd.read_csv(
        path,
        usecols=selected,
        dtype={"item_id": "string"},
        na_values=["", "null", "None"],
        keep_default_na=True,
        low_memory=False,
    )
    records: list[ProductRecord] = []
    seen_ids: set[str] = set()
    for values in frame.itertuples(index=False, name=None):
        row = {
            column: None if pd.isna(value) else value
            for column, value in zip(frame.columns, values, strict=True)
        }
        record = normalize_product(row)
        if not record.item_id or not record.raw_name:
            continue
        if record.item_id in seen_ids:
            continue
        seen_ids.add(record.item_id)
        records.append(record)

    return records
