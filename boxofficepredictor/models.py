from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingRecord:
    revenue: float
    title: str
    budget: float
    twitter_score: float
    youtube_ratio: float
    youtube_like_count: float


@dataclass(frozen=True)
class PredictionRequest:
    title: str
    budget: float
    twitter_score: float
    youtube_ratio: float | None = None
    youtube_like_count: float | None = None


@dataclass(frozen=True)
class SentimentReport:
    score: float
    label: str
    tweet_count: int
    positive_hits: int
    negative_hits: int


@dataclass(frozen=True)
class PredictionResult:
    title: str
    estimated_box_office: float
    training_sample_count: int
    training_rmse_log: float
    twitter_score: float
    budget: float
    youtube_ratio: float
    youtube_like_count: float
