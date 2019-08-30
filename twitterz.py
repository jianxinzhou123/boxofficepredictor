import tweepy
import sys
import re

CONSUMER_KEY = 'RLLdkysFGMypApj6MKJEUuRSZ'
CONSUMER_SECRET = 'ZSVU6YMkVl0b9eKRZktlbf3ztV5ImyCmouLtCieVP4JIwgrUG3'
ACCESS_TOKEN = '1038443235155353600-iRl55iDNrNlxBFTmD6hoxLUwchL6Uy'
ACCESS_TOKEN_SECRET = 'qAmFlhK0VgnDx1X0yXC1pXVPmKxNHADnBsilRihsGdCEd'
dataset = []

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def fetchTweets(topic, count):
        
    for tweet in tweepy.Cursor(api.search, q=topic +' -filter:retweets AND  -filter:replies AND -filter:links', tweet_mode='extended', rpp=100, lang='en').items(count):
        tweet.full_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split())
        dataset.append(tweet.full_text)

    return dataset

    
        
    
if __name__ == '__main__':
    print("Please run the correct program! It's called run_predictor.py!")
