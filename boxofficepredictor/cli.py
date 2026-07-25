from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .collector import configure_parser as configure_collect_parser
from .collector import main as collect_main
from .data_prep import deduplicate_dataset_rows
from .dataset import write_prediction_request
from .models import PredictionRequest
from .paths import PREDICTION_INPUT_PATH, RAW_DATASET_PATH, TRAINING_DATA_PATH
from .predictor import BoxOfficePredictorService
from .sentiment import analyze_tweets
from .twitter import FileTweetProvider, StaticTweetProvider, TwitterRecentSearchProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a movie's box office from budget and Twitter sentiment."
    )
    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser("predict", help="Run a box office prediction")
    predict_parser.add_argument("--title", help="Movie title")
    predict_parser.add_argument("--budget", type=float, help="Estimated production budget in USD")
    predict_parser.add_argument("--twitter-score", type=float, help="Precomputed Twitter sentiment score in the -1..1 range")
    predict_parser.add_argument("--topic", help="Hashtag or search topic for live Twitter collection")
    predict_parser.add_argument("--tweet", action="append", default=[], help="Inline tweet text; repeat to provide more than one")
    predict_parser.add_argument("--tweets-file", help="Path to a .txt or .csv file containing tweets")
    predict_parser.add_argument("--tweet-limit", type=int, default=100, help="Maximum number of tweets to analyze")
    predict_parser.add_argument("--youtube-ratio", type=float, help="Optional YouTube like ratio")
    predict_parser.add_argument("--youtube-like-count", type=float, help="Optional YouTube like count")
    predict_parser.add_argument("--training-data", default=str(TRAINING_DATA_PATH))
    predict_parser.add_argument("--write-predict-csv", default=str(PREDICTION_INPUT_PATH), help="Write the normalized input row to a CSV file")
    predict_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")

    collect_parser = subparsers.add_parser("collect", help="Append a labeled row to the raw dataset")
    configure_collect_parser(collect_parser)

    prepare_parser = subparsers.add_parser("prepare-data", help="Deduplicate the raw dataset into train.csv")
    prepare_parser.add_argument("--input", default=str(RAW_DATASET_PATH))
    prepare_parser.add_argument("--output", default=str(TRAINING_DATA_PATH))

    return parser


def _prompt_text(label: str) -> str:
    value = ""
    while not value.strip():
        value = input(f"{label}: ").strip()
    return value


def _prompt_float(label: str) -> float:
    while True:
        try:
            return float(_prompt_text(label))
        except ValueError:
            print("Enter a numeric value.")


def _resolve_twitter_score(args: argparse.Namespace) -> tuple[float, dict[str, object] | None]:
    if args.twitter_score is not None:
        return args.twitter_score, None

    if args.tweets_file:
        provider = FileTweetProvider(Path(args.tweets_file))
        tweets = provider.fetch_tweets(args.title or "", args.tweet_limit)
    elif args.tweet:
        provider = StaticTweetProvider(args.tweet)
        tweets = provider.fetch_tweets(args.title or "", args.tweet_limit)
    elif args.topic:
        provider = TwitterRecentSearchProvider.from_env()
        tweets = provider.fetch_tweets(args.topic, args.tweet_limit)
    else:
        score = _prompt_float("Twitter sentiment score (-1 to 1), or Ctrl+C to exit")
        return score, None

    report = analyze_tweets(tweets)
    return report.score, {
        "label": report.label,
        "tweet_count": report.tweet_count,
        "positive_hits": report.positive_hits,
        "negative_hits": report.negative_hits,
    }


def _ensure_inputs(args: argparse.Namespace) -> None:
    if not args.title:
        args.title = _prompt_text("Movie title")
    if args.budget is None:
        args.budget = _prompt_float("Estimated production budget in USD")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    if normalized_argv and normalized_argv[0] not in {"predict", "collect", "prepare-data", "-h", "--help"}:
        normalized_argv = ["predict", *normalized_argv]

    args = parser.parse_args(normalized_argv)
    if not args.command:
        args = parser.parse_args(["predict", *normalized_argv])

    if args.command == "collect":
        return collect_main(normalized_argv[1:])

    if args.command == "prepare-data":
        count = deduplicate_dataset_rows(Path(args.input), Path(args.output))
        print(f"Prepared {count} unique rows in {args.output}")
        return 0

    _ensure_inputs(args)

    twitter_score, sentiment_metadata = _resolve_twitter_score(args)
    request = PredictionRequest(
        title=args.title,
        budget=args.budget,
        twitter_score=twitter_score,
        youtube_ratio=args.youtube_ratio,
        youtube_like_count=args.youtube_like_count,
    )

    service = BoxOfficePredictorService.from_csv(Path(args.training_data))
    result = service.predict(request)

    if args.write_predict_csv:
        write_prediction_request(Path(args.write_predict_csv), request)

    if args.json:
        print(
            json.dumps(
                {
                    "title": result.title,
                    "estimated_box_office": result.estimated_box_office,
                    "training_sample_count": result.training_sample_count,
                    "training_rmse_log": result.training_rmse_log,
                    "twitter_score": result.twitter_score,
                    "budget": result.budget,
                    "youtube_ratio": result.youtube_ratio,
                    "youtube_like_count": result.youtube_like_count,
                    "sentiment": sentiment_metadata,
                },
                indent=2,
            )
        )
        return 0

    if sentiment_metadata is not None:
        print(
            f"Analyzed {sentiment_metadata['tweet_count']} tweets. "
            f"Sentiment is {sentiment_metadata['label']} with score {twitter_score:.4f}."
        )

    print(f"Predicted worldwide box office for {result.title}: ${result.estimated_box_office:,.2f}")
    print(f"Training samples: {result.training_sample_count} | log-RMSE: {result.training_rmse_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
