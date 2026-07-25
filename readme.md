# Box Office Predictor

This project has been refactored into a small, offline-friendly Python package that predicts box office revenue from:

- estimated production budget
- Twitter sentiment score or tweet text samples
- optional YouTube engagement inputs

The runtime no longer depends on the old Keras, scikit-learn, Tweepy, or NLTK stack, which was pinned to versions that no longer run cleanly on current Python builds.

## Project layout

```text
boxofficepredictor/
├── boxofficepredictor/    # package code and CLI
├── data/                  # raw, prepared, and last-run CSV inputs
├── docs/                  # non-code project documents
├── tests/                 # unit tests
├── readme.md
└── requirements.txt
```

## What changed

- prediction logic now lives in a package: `boxofficepredictor/`
- sentiment analysis is local and rule-based instead of relying on NLTK downloads
- tweet collection is pluggable: inline tweets, CSV/text files, or optional live Twitter API v2 search through `TWITTER_BEARER_TOKEN`
- training data collection and dataset preparation are exposed as package subcommands
- legacy one-off scripts were removed from the root
- committed API secrets were removed from the runtime path

## Quick start

Run with a precomputed sentiment score:

```bash
python3 -m boxofficepredictor predict \
    --title "Example Movie" \
    --budget 120000000 \
    --twitter-score 0.35
```

Run from tweet text samples in a file:

```bash
python3 -m boxofficepredictor predict \
    --title "Example Movie" \
    --budget 120000000 \
    --tweets-file tweets.txt
```

Run from inline tweets:

```bash
python3 -m boxofficepredictor predict \
    --title "Example Movie" \
    --budget 120000000 \
    --tweet "This trailer looks amazing and the cast is great" \
    --tweet "Absolutely hyped for opening weekend"
```

Emit JSON for automation:

```bash
python3 -m boxofficepredictor predict \
    --title "Example Movie" \
    --budget 120000000 \
    --twitter-score 0.35 \
    --json
```

## Live Twitter collection

The project can query the Twitter/X recent search API if you provide a bearer token:

```bash
export TWITTER_BEARER_TOKEN="..."
python3 -m boxofficepredictor predict \
    --title "Example Movie" \
    --budget 120000000 \
    --topic "#ExampleMovie"
```

If you do not have API access, use `--tweet`, `--tweets-file`, or `--twitter-score`.

## Prepare training data

Rebuild the prepared training dataset from the raw dataset:

```bash
python3 -m boxofficepredictor prepare-data
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## Add training rows

Append a labeled movie row to `data/dataset.csv`:

```bash
python3 -m boxofficepredictor collect \
    --title "Example Movie" \
    --budget 120000000 \
    --revenue 450000000 \
    --tweet "This trailer looks amazing" \
    --tweet "Opening weekend hype is strong"
```

To append and rebuild `data/train.csv` in one step:

```bash
python3 -m boxofficepredictor collect \
    --title "Example Movie" \
    --budget 120000000 \
    --revenue 450000000 \
    --twitter-score 0.35 \
    --sync-train
```
