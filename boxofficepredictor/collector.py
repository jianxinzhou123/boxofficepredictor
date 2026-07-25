from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .data_prep import deduplicate_dataset_rows
from .models import TrainingRecord
from .paths import RAW_DATASET_PATH, TRAINING_DATA_PATH
from .sentiment import analyze_tweets
from .twitter import FileTweetProvider, StaticTweetProvider, TwitterRecentSearchProvider


TRAINING_HEADER = [
    "Total Box Office",
    "Movie Name",
    "Initial Budget",
    "TwitterSense True Score",
    "Youtube Ratio Score",
    "Youtube Trailer Like Count",
]


def append_training_record(csv_path: Path, record: TrainingRecord) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(TRAINING_HEADER)
        writer.writerow(
            [
                record.revenue,
                record.title,
                record.budget,
                record.twitter_score,
                record.youtube_ratio,
                record.youtube_like_count,
            ]
        )


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


def _resolve_twitter_score(args: argparse.Namespace) -> float:
    if args.twitter_score is not None:
        return args.twitter_score
    if args.tweets_file:
        tweets = FileTweetProvider(Path(args.tweets_file)).fetch_tweets(args.title, args.tweet_limit)
        return analyze_tweets(tweets).score
    if args.tweet:
        tweets = StaticTweetProvider(args.tweet).fetch_tweets(args.title, args.tweet_limit)
        return analyze_tweets(tweets).score
    if args.topic:
        tweets = TwitterRecentSearchProvider.from_env().fetch_tweets(args.topic, args.tweet_limit)
        return analyze_tweets(tweets).score
    return _prompt_float("Twitter sentiment score (-1 to 1)")


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--title")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--revenue", type=float)
    parser.add_argument("--twitter-score", type=float)
    parser.add_argument("--tweet", action="append", default=[])
    parser.add_argument("--tweets-file")
    parser.add_argument("--topic")
    parser.add_argument("--tweet-limit", type=int, default=100)
    parser.add_argument("--youtube-ratio", type=float, default=0.0)
    parser.add_argument("--youtube-like-count", type=float, default=0.0)
    parser.add_argument("--output", default=str(RAW_DATASET_PATH))
    parser.add_argument("--sync-train", action="store_true", help="Also rebuild the deduplicated training dataset")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append a labeled training sample to the raw dataset")
    return configure_parser(parser)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    title = args.title or _prompt_text("Movie title")
    budget = args.budget if args.budget is not None else _prompt_float("Production budget in USD")
    revenue = args.revenue if args.revenue is not None else _prompt_float("Actual box office revenue in USD")
    twitter_score = _resolve_twitter_score(args)

    record = TrainingRecord(
        revenue=revenue,
        title=title,
        budget=budget,
        twitter_score=twitter_score,
        youtube_ratio=args.youtube_ratio,
        youtube_like_count=args.youtube_like_count,
    )
    append_training_record(Path(args.output), record)
    print(f"Saved training sample for {title} to {args.output}")
    if args.sync_train:
        count = deduplicate_dataset_rows(Path(args.output), TRAINING_DATA_PATH)
        print(f"Rebuilt {TRAINING_DATA_PATH} with {count} unique movies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
