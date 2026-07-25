from __future__ import annotations

import csv
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol


class TweetProvider(Protocol):
    def fetch_tweets(self, topic: str, limit: int) -> list[str]:
        ...


class StaticTweetProvider:
    def __init__(self, tweets: list[str]) -> None:
        self._tweets = tweets

    def fetch_tweets(self, topic: str, limit: int) -> list[str]:
        del topic
        return self._tweets[:limit]


class FileTweetProvider:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def fetch_tweets(self, topic: str, limit: int) -> list[str]:
        del topic
        suffix = self.file_path.suffix.lower()
        if suffix == ".csv":
            return self._read_csv(limit)
        return self._read_text(limit)

    def _read_csv(self, limit: int) -> list[str]:
        with self.file_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            field_names = reader.fieldnames or []
            preferred = ["tweet", "text", "content", "body", "message"]
            selected = next((name for name in preferred if name in field_names), None)
            if selected is None:
                selected = field_names[0] if field_names else None
            if selected is None:
                return []
            return [row.get(selected, "") for row in list(reader)[:limit]]

    def _read_text(self, limit: int) -> list[str]:
        with self.file_path.open(encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        return lines[:limit]


class TwitterRecentSearchProvider:
    def __init__(self, bearer_token: str) -> None:
        self.bearer_token = bearer_token

    @classmethod
    def from_env(cls) -> "TwitterRecentSearchProvider":
        bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            raise RuntimeError(
                "TWITTER_BEARER_TOKEN is not set. Use --tweets-file, --tweet, or --twitter-score instead."
            )
        return cls(bearer_token)

    def fetch_tweets(self, topic: str, limit: int) -> list[str]:
        if not topic.strip():
            raise ValueError("A topic or hashtag is required for live Twitter collection")

        max_results = min(max(limit, 10), 100)
        query = f"{topic} lang:en -is:retweet -is:reply has:text"
        params = urllib.parse.urlencode(
            {
                "query": query,
                "max_results": max_results,
                "tweet.fields": "lang,text",
            }
        )
        request = urllib.request.Request(
            f"https://api.twitter.com/2/tweets/search/recent?{params}",
            headers={"Authorization": f"Bearer {self.bearer_token}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        tweets = [entry.get("text", "") for entry in payload.get("data", [])]
        if not tweets:
            raise RuntimeError("Twitter returned no tweets for that topic")
        return tweets[:limit]
