from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout
from io import StringIO

from boxofficepredictor.cli import main as cli_main
from boxofficepredictor.models import PredictionRequest
from boxofficepredictor.paths import RAW_DATASET_PATH, TRAINING_DATA_PATH
from boxofficepredictor.predictor import BoxOfficePredictorService
from boxofficepredictor.sentiment import analyze_tweets


class SentimentTests(unittest.TestCase):
    def test_sentiment_prefers_positive_language(self) -> None:
        report = analyze_tweets(
            [
                "This movie looks amazing and absolutely fantastic.",
                "The trailer is awesome and I love the cast.",
            ]
        )
        self.assertGreater(report.score, 0.15)
        self.assertEqual(report.label, "positive")

    def test_sentiment_flags_negative_language(self) -> None:
        report = analyze_tweets(
            [
                "The trailer looks boring and terrible.",
                "This feels like a weak and disappointing flop.",
            ]
        )
        self.assertLess(report.score, -0.15)
        self.assertEqual(report.label, "negative")


class PredictorTests(unittest.TestCase):
    def test_prediction_returns_positive_value(self) -> None:
        service = BoxOfficePredictorService.from_csv(TRAINING_DATA_PATH)
        result = service.predict(
            PredictionRequest(
                title="Example Movie",
                budget=120000000,
                twitter_score=0.35,
            )
        )
        self.assertGreater(result.estimated_box_office, 0.0)
        self.assertEqual(result.title, "Example Movie")

    def test_feature_vector_prioritizes_youtube_ratio_over_twitter(self) -> None:
        service = BoxOfficePredictorService.from_csv(TRAINING_DATA_PATH)
        request = PredictionRequest(
            title="Priority Check",
            budget=120000000,
            twitter_score=0.4,
            youtube_ratio=0.9,
            youtube_like_count=500000,
        )

        features = service._feature_vector(request)
        self.assertEqual(
            features[2],
            request.youtube_ratio * BoxOfficePredictorService.YOUTUBE_RATIO_WEIGHT,
        )
        self.assertEqual(
            features[4],
            request.twitter_score * BoxOfficePredictorService.TWITTER_SCORE_WEIGHT,
        )
        self.assertLess(features[4], features[2])


class DataLayoutTests(unittest.TestCase):
    def test_default_training_data_exists(self) -> None:
        self.assertTrue(Path(TRAINING_DATA_PATH).exists())


class CliTests(unittest.TestCase):
    def test_predict_subcommand_emits_json(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "predict.csv"
            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = cli_main(
                    [
                        "predict",
                        "--title",
                        "Example Movie",
                        "--budget",
                        "120000000",
                        "--tweet",
                        "This trailer looks amazing and the cast is great",
                        "--tweet",
                        "Absolutely hyped for opening weekend",
                        "--write-predict-csv",
                        str(output_path),
                        "--json",
                    ]
                )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["title"], "Example Movie")
            self.assertTrue(output_path.exists())

    def test_predict_computes_youtube_ratio_and_defaults_twitter_score(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = cli_main(
                [
                    "predict",
                    "--title",
                    "Example Movie",
                    "--budget",
                    "120000000",
                    "--youtube-like-count",
                    "900",
                    "--youtube-dislike-count",
                    "100",
                    "--write-predict-csv",
                    "",
                    "--json",
                ]
            )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["twitter_score"], 0.0)
        self.assertAlmostEqual(payload["youtube_ratio"], 0.9, places=6)
        self.assertEqual(payload["youtube_like_count"], 900.0)

    def test_prepare_data_subcommand_deduplicates_rows(self) -> None:
        with TemporaryDirectory() as directory:
            raw_path = Path(directory) / "dataset.csv"
            train_path = Path(directory) / "train.csv"
            raw_path.write_text(
                "Total Box Office,Movie Name,Initial Budget,TwitterSense True Score,Youtube Ratio Score,Youtube Trailer Like Count\n"
                "1,Movie A,10,0.1,0.9,100\n"
                "2,Movie A,11,0.2,0.8,120\n"
                "3,Movie B,12,0.3,0.7,140\n",
                encoding="utf-8",
            )

            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = cli_main(
                    [
                        "prepare-data",
                        "--input",
                        str(raw_path),
                        "--output",
                        str(train_path),
                    ]
                )

            rows = list(csv.DictReader(train_path.open(encoding="utf-8")))
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(rows), 2)

    def test_collect_subcommand_writes_dataset_row(self) -> None:
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.csv"
            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = cli_main(
                    [
                        "collect",
                        "--title",
                        "Example Movie",
                        "--budget",
                        "120000000",
                        "--revenue",
                        "450000000",
                        "--youtube-like-count",
                        "900",
                        "--youtube-dislike-count",
                        "100",
                        "--output",
                        str(dataset_path),
                    ]
                )

            rows = list(csv.DictReader(dataset_path.open(encoding="utf-8")))
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Movie Name"], "Example Movie")
            self.assertEqual(rows[0]["TwitterSense True Score"], "0.0")
            self.assertEqual(rows[0]["Youtube Trailer Like Count"], "900.0")
            self.assertAlmostEqual(float(rows[0]["Youtube Ratio Score"]), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
