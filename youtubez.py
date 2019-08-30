from googleapiclient.discovery import build
import argparse
import csv
import unidecode
import json
import requests
import logging

DEVELOPER_KEY = "AIzaSyCsnLAeRt9pm6V4FU5W-g-KxPMw3mdOk7E"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


api = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY, cache_discovery=False)
logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)



def getLikeCount(url):
    aggregated_results = api.videos().list(id=url, part="statistics").execute()
    for singleton in aggregated_results.get("items", []):
        if 'likeCount' not in singleton["statistics"]:
            likeCount = 0
        else:
            likeCount = singleton["statistics"]['likeCount']

    return float(likeCount)


def getDislikeCount(url):
    aggregated_results = api.videos().list(id=url, part="statistics").execute()
    for singleton in aggregated_results.get("items", []):
        if 'dislikeCount' not in singleton["statistics"]:
            dislikeCount = 0
        else:
            dislikeCount = singleton["statistics"]['dislikeCount']

    return float(dislikeCount)


def getTotalViewCount(url):
    aggregated_results = api.videos().list(id=url, part="statistics").execute()
    for singleton in aggregated_results.get("items", []):
        if 'viewCount' not in singleton["statistics"]:
            viewCount = 0
        else:
            viewCount = singleton["statistics"]['viewCount']

    return float(viewCount)


if __name__ == '__main__':
    print("Please run the correct program! It's called run_predictor.py!")
