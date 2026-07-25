from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

from .models import PredictionRequest, TrainingRecord


TRAINING_HEADERS = [
    "Total Box Office",
    "Movie Name",
    "Initial Budget",
    "TwitterSense True Score",
    "Youtube Ratio Score",
    "Youtube Trailer Like Count",
]


def _to_float(value: str | None, *, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def load_training_records(csv_path: Path) -> list[TrainingRecord]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = [
            TrainingRecord(
                revenue=_to_float(row.get("Total Box Office")),
                title=(row.get("Movie Name") or "Unknown").strip(),
                budget=_to_float(row.get("Initial Budget")),
                twitter_score=_to_float(row.get("TwitterSense True Score")),
                youtube_ratio=_to_float(row.get("Youtube Ratio Score")),
                youtube_like_count=_to_float(row.get("Youtube Trailer Like Count")),
            )
            for row in reader
        ]

    if not records:
        raise ValueError(f"No training rows found in {csv_path}")

    return records


def read_prediction_request(csv_path: Path) -> PredictionRequest:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No prediction rows found in {csv_path}")

    row = rows[0]
    return PredictionRequest(
        title=(row.get("Movie Name") or "Unknown").strip(),
        budget=_to_float(row.get("Initial Budget")),
        twitter_score=_to_float(row.get("TwitterSense True Score")),
        youtube_ratio=_to_float(row.get("Youtube Ratio Score"), default=0.0),
        youtube_like_count=_to_float(row.get("Youtube Trailer Like Count"), default=0.0),
    )


def write_prediction_request(csv_path: Path, request: PredictionRequest) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRAINING_HEADERS)
        writer.writerow(
            [
                "",
                request.title,
                request.budget,
                request.twitter_score,
                request.youtube_ratio if request.youtube_ratio is not None else "",
                request.youtube_like_count if request.youtube_like_count is not None else "",
            ]
        )


def training_defaults(records: list[TrainingRecord]) -> dict[str, float]:
    return {
        "youtube_ratio": median(record.youtube_ratio for record in records),
        "youtube_like_count": median(record.youtube_like_count for record in records),
    }
