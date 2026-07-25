from __future__ import annotations

import math
import re

from .models import SentimentReport


TOKEN_RE = re.compile(r"[a-z']+")
NEGATIONS = {"not", "never", "no", "hardly", "barely", "isn't", "wasn't", "don't", "didn't"}
BOOSTERS = {"very", "really", "so", "extremely", "super", "absolutely", "totally"}
POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "best", "blockbuster", "buzzing", "captivating",
    "classic", "cool", "electric", "epic", "excellent", "fantastic", "fun", "good",
    "great", "hype", "hyped", "iconic", "impressive", "incredible", "love", "loved",
    "masterpiece", "mustsee", "mustwatch", "outstanding", "phenomenal", "positive",
    "smart", "solid", "spectacular", "strong", "stunning", "success", "thrilling",
    "unmissable", "victory", "wow",
}
NEGATIVE_WORDS = {
    "awful", "bad", "boring", "cheap", "confusing", "cringe", "cringey", "dated",
    "disappointing", "dull", "flop", "forgettable", "frustrating", "hate", "hated",
    "horrible", "lifeless", "mess", "negative", "predictable", "skip", "slow",
    "terrible", "tired", "unfunny", "weak", "worse", "worst",
}


def _tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("must see", "mustsee").replace("must watch", "mustwatch")
    return TOKEN_RE.findall(normalized)


def score_text(text: str) -> tuple[float, int, int]:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0, 0, 0

    sentiment_total = 0.0
    positive_hits = 0
    negative_hits = 0

    for index, token in enumerate(tokens):
        base = 0.0
        if token in POSITIVE_WORDS:
            base = 1.0
            positive_hits += 1
        elif token in NEGATIVE_WORDS:
            base = -1.0
            negative_hits += 1

        if base == 0.0:
            continue

        if index > 0 and tokens[index - 1] in BOOSTERS:
            base *= 1.5
        if index > 0 and tokens[index - 1] in NEGATIONS:
            base *= -1.0
        if index > 1 and tokens[index - 2] in NEGATIONS:
            base *= -1.0

        sentiment_total += base

    if sentiment_total == 0.0:
        return 0.0, positive_hits, negative_hits

    normalized = sentiment_total / max(1.0, math.sqrt(len(tokens)))
    return max(-1.0, min(1.0, normalized / 2.0)), positive_hits, negative_hits


def analyze_tweets(tweets: list[str]) -> SentimentReport:
    cleaned = [tweet.strip() for tweet in tweets if tweet and tweet.strip()]
    if not cleaned:
        raise ValueError("At least one tweet or tweet-like text sample is required")

    scores: list[float] = []
    positive_hits = 0
    negative_hits = 0
    for tweet in cleaned:
        score, positive, negative = score_text(tweet)
        scores.append(score)
        positive_hits += positive
        negative_hits += negative

    average_score = sum(scores) / len(scores)
    if average_score > 0.15:
        label = "positive"
    elif average_score < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return SentimentReport(
        score=round(average_score, 6),
        label=label,
        tweet_count=len(cleaned),
        positive_hits=positive_hits,
        negative_hits=negative_hits,
    )
