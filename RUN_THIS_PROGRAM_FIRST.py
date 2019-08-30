##BoxOffice Predictor 0.1.0 Copyright 2019
##Created by:
##    Jian (Sam) Zhou         Andrew Chasin
##    Peter Cougan            John Lauer
##    Jacquelyn Brozyna
##Project misuse witout authorization is prohibitted.

import twitter_sentiment_analyzerz
import neural_network_model
import Engine
import twitterz
import youtubez
import re
import datetime
import requests
import csv
import RemoveDuplicateCSVRows
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse, parse_qs
from tmdbv3api import TMDb
from tmdbv3api import Movie


def main():
    RemoveDuplicateCSVRows.remove()
    title, intial_budget, vote_average = RetriveIntialData()
    twitter_true_score = TwitterComponent()
    youtube_like_to_dislike_ratio, youtube_likeCount, youtube_total_view = YoutubeComponent()
    learning(title, intial_budget, twitter_true_score, youtube_like_to_dislike_ratio, youtube_likeCount)


def get_id(url):
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]

alreadyRevenued = False

def RetriveIntialData():
    global alreadyRevenued
    error_predict = "This movie has already reported it's box office revenue! See below."
    guard3 = False;
    while(guard3==False):

        try:
            movieName = input("\nWhat is the name of the movie to predict?")
            tmdb = TMDb()
            tmdb.api_key = "10163a322f4558e7cc6411377702d81d"

            movie = Movie()
            
            search = movie.search(movieName)

            if search is not None:
                for e in search:
                    movieID= e.id
                    break
                else:
                    movieID=None


            m= movie.details(movieID)

            initial_budget = m.budget
            revenue = m.revenue
            vote_average = m.vote_average
            status = m.status
            language = m.original_language.upper()
            title = m.original_title
            company = m.production_companies[0]['name']
            poster_path = m.poster_path
            poster = "http://image.tmdb.org/t/p/w300/" + poster_path
            
            if (revenue is not 0):
                alreadyRevenued=True
                print(error_predict)
            else:
                error_predict = "This movie has yet reported it's box office revenue, we will attempt to predict it via machine learning."
                alreadyRevenued=False
                print(error_predict)

            print("\n===============================================================")
            print("Fetching information... Please wait...")
            print("-Full Movie Name: " + title)
            print("-Production Company: " + company)
            print("-Original Language: " + language)
            print("-Status: " + status)

            try:
                response = requests.get(poster)
                img = Image.open(BytesIO(response.content))
                img.show()
            except:
                print("RENDERING EXPCETION: Cannot render movie poster...")
            
            if(initial_budget is not 0):
                print("-Intial Budget: " + (str(int(initial_budget))))
                
            else:
                initial_budget_string = ''
                while not re.match('^-?[0-9]*\.?[0-9]+$',initial_budget_string):
                    initial_budget_string = input("-***Initial Budget: Not yet reported. Enter a valid estimate.***")

                initial_budget = int(str(initial_budget_string))
            
            if(alreadyRevenued==True):
                print("-Revenue: " + (str(int(revenue))))
                print("-To-Date Rating: " + str(float(vote_average)))
                guard3=False
                print("===============================================================")
                print("Try another movie?")
                    
            else:
                print("-***Revenue: Not yet determined or reported.***")
                guard3=True
                print("===============================================================")
            

        except:
            print("Failed to retrieve information. Make sure this movie actually exists!")

    return title, initial_budget, vote_average

    
def TwitterComponent():
    guard1 = False
    twitter_true_score = 0.0
    tweet_count = 1500
    global alreadyRevenued
    while(guard1==False):

        if alreadyRevenued==True:
            break

        try:
            topic = input("\nEnter the associated OFFICIAL movie hashtag without the pound sign and spaces.")


        except:
            print("\nYou've entered bad information. Try again.")
            
        
        try:
            print("\n---------------------------------------------------------------")
            print("Analzying " + str(int(tweet_count)) + " fetched datas... This may take a while depending on your sample size. Please wait...")
            datas = twitter_sentiment_analyzerz.getData(topic, tweet_count)
            twitter_true_score=twitter_sentiment_analyzerz.trueScoreEval(datas)
            guard1=True
            print("---------------------------------------------------------------")

        except:
            print("Failed to retrieve information. Make sure you've entered a VALID TWITTER hashtag.")


    return twitter_true_score


def YoutubeComponent():
    guard2= False
    youtube_like_to_dislike_ratio = 0.0
    global alreadyRevenued
    while(guard2==False):
        
        if alreadyRevenued==True:
            break
        
        youtubeURL = input("\nWhat is the URL of the corresponding OFFICIAL movie trailer on YouTube?")
        youtubeURL = get_id(youtubeURL)

        try:
            youtube_likeCount= youtubez.getLikeCount(youtubeURL)
            youtube_dislikeCount = youtubez.getDislikeCount(youtubeURL)
            youtube_total_view = youtubez.getTotalViewCount(youtubeURL)
            try:
                youtube_like_to_dislike_ratio = youtube_likeCount / (youtube_likeCount + youtube_dislikeCount)
                print("\n---------------------------------------------------------------")
                print("Parsing... This may take a while depending on your sample size. Please wait...")
                print("-Total Like Count: " + (str(float(youtube_likeCount))))
                print("-Total Disike Count: " + (str(float(youtube_dislikeCount))))
                print("-Total View: " + (str(float(youtube_total_view))))
                print("-The Youtube Like to Dislike Ratio on this video is: " + (str(float(youtube_like_to_dislike_ratio))))
                guard2=True
                print("---------------------------------------------------------------")
                
            except:
                print("The like count has been obfuscated by the user. Try another trailer instead!")
                guard2=False

        except:
            print("Failed to retrieve information. Make sure you've entered a VALID YOUTUBE URL address and ALL assets of the video are public.")

    return youtube_like_to_dislike_ratio, youtube_likeCount, youtube_total_view

def learning(title, budget, tTrueScore, yRatio, yLikeCount):
    if(alreadyRevenued==False):
        print("\n===============================================================")
        print("Fetching mined data into the machine learning model... Please wait...")
        print("Initial Budget: " + str(int(budget)))
        print("TwitterSense True Score: " + str(float(tTrueScore)) + " as of the current fetch batch.")
        print("Youtube Like-to-Dislike Ratio: " + str(float(yRatio)))
        print("Youtube Like Count: " + str(float(yLikeCount)))
        print("===============================================================")

        try:
            with open('Predict.csv', 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(['Total Box Office', 'Movie Name', 'Initial Budget', 'TwitterSense True Score', 'Youtube Ratio Score', 'Youtube Trailer Like Count'])
                filewriter.writerow([None, title, budget, tTrueScore, yRatio, yLikeCount])
            csvfile.close()
            
            try:
                predict_budget = Engine.predict()
            except:
                neural_network_model.CreateModel()
                predict_budget = Engine.predict()

            print("\nBased on our machine learning model from the dataset we've gathered, the estimated revenue of " + title + " would be: $" + str(float(predict_budget)) + ".")

        except:
            print("\nA fatal error occured while predicting!")

        return predict_budget

    else:
        pass
    
    
if __name__ == '__main__':
    main()






