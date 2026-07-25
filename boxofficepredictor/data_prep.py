from __future__ import annotations

import csv
from pathlib import Path

from .dataset import TRAINING_HEADERS


def deduplicate_dataset_rows(input_path: Path, output_path: Path) -> int:
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Dataset file is empty: {input_path}")

    header = rows[0]
    body = rows[1:]
    deduplicated_rows: list[list[str]] = []
    seen_titles: set[str] = set()

    for row in body:
        if not any(field.strip() for field in row):
            continue
        if len(row) < 2:
            continue
        title = row[1].strip().lower()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        deduplicated_rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or TRAINING_HEADERS)
        writer.writerows(deduplicated_rows)

    return len(deduplicated_rows)
