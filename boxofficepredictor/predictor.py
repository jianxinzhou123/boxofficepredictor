from __future__ import annotations

import math

from .dataset import load_training_records, training_defaults
from .models import PredictionRequest, PredictionResult, TrainingRecord
from .paths import TRAINING_DATA_PATH
from .regression import RidgeRegressor


class BoxOfficePredictorService:
    # Favor YouTube engagement ratio as the primary social signal.
    YOUTUBE_RATIO_WEIGHT = 1.0
    TWITTER_SCORE_WEIGHT = 0.35

    def __init__(self, records: list[TrainingRecord], *, regularization: float = 2.0) -> None:
        self.records = records
        self.defaults = training_defaults(records)
        self.model = RidgeRegressor(regularization=regularization)
        self._fit()

    @classmethod
    def from_csv(cls, csv_path=TRAINING_DATA_PATH) -> "BoxOfficePredictorService":
        return cls(load_training_records(csv_path))

    def _feature_vector(self, request: PredictionRequest) -> list[float]:
        youtube_ratio = (
            request.youtube_ratio
            if request.youtube_ratio is not None
            else self.defaults["youtube_ratio"]
        )
        youtube_like_count = (
            request.youtube_like_count
            if request.youtube_like_count is not None
            else self.defaults["youtube_like_count"]
        )
        return [
            1.0,
            math.log1p(max(0.0, request.budget)),
            youtube_ratio * self.YOUTUBE_RATIO_WEIGHT,
            math.log1p(max(0.0, youtube_like_count)),
            request.twitter_score * self.TWITTER_SCORE_WEIGHT,
        ]

    def _fit(self) -> None:
        samples = [
            self._feature_vector(
                PredictionRequest(
                    title=record.title,
                    budget=record.budget,
                    twitter_score=record.twitter_score,
                    youtube_ratio=record.youtube_ratio,
                    youtube_like_count=record.youtube_like_count,
                )
            )
            for record in self.records
        ]
        targets = [math.log1p(record.revenue) for record in self.records]
        self.model.fit(samples, targets)
        self.training_rmse_log = self.model.rmse(samples, targets)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        features = self._feature_vector(request)
        estimated_box_office = math.expm1(self.model.predict(features))
        youtube_ratio = request.youtube_ratio if request.youtube_ratio is not None else self.defaults["youtube_ratio"]
        youtube_like_count = (
            request.youtube_like_count
            if request.youtube_like_count is not None
            else self.defaults["youtube_like_count"]
        )
        return PredictionResult(
            title=request.title,
            estimated_box_office=round(max(0.0, estimated_box_office), 2),
            training_sample_count=len(self.records),
            training_rmse_log=round(self.training_rmse_log, 4),
            twitter_score=request.twitter_score,
            budget=request.budget,
            youtube_ratio=youtube_ratio,
            youtube_like_count=youtube_like_count,
        )
