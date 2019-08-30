import RUN_THIS_PROGRAM_FIRST
from tmdbv3api import TMDb
from tmdbv3api import Movie
import csv
import sys
import twitter_sentiment_analyzerz
import youtubez
from urllib.parse import urlparse, parse_qs

firstTimeWrite = True
alreadyRevenued=False
    

def init():
    movieName = input("\nMovie Name?")
    tmdb = TMDb()
    tmdb.api_key = "10163a322f4558e7cc6411377702d81d"

    movie = Movie()

    try:
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
        title = m.original_title

        print("-Full name: " + title)

        if(initial_budget is not 0):
            print("-Intial Budget: " + (str(int(initial_budget))))

        else:
             print("NO INITIAL BUDGET REPORTED. DO NOT USE THIS ENTRY FOR TRAINING DATA.")


        if(revenue is not 0):
            print("-Revenue: " + (str(int(revenue))))

        else:
            print("NO REVENUE REPORTED. DO NOT USE THIS ENTRY FOR TRAINING DATA.")
            exit()
    except:
        print("\nBad movie entry?")


    return title, initial_budget, revenue

def get_id(url):
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]


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
            truescore=twitter_sentiment_analyzerz.trueScoreEval(datas)
            guard1=True
            print("---------------------------------------------------------------")

        except:
            print("Failed to retrieve information. Make sure you've entered a VALID TWITTER hashtag.")


    return truescore



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

def program():
    global firstTimeWrite
    stopString = ""
    stop = False
    while(stop==False):
        try:
            title, initial_budget, revenue = init()
            truescore = TwitterComponent()
            youtube_like_to_dislike_ratio, youtube_likeCount, youtube_total_view = YoutubeComponent()
        except:
            print("Possibly loss of data. Please double check privacy settings to ensure all API information is public.")

        try:
            with open('Dataset.csv', 'a') as csvfile:
                if(firstTimeWrite==True):
                    filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewriter.writerow([]) 
                    filewriter.writerow([revenue, title, initial_budget, truescore, youtube_like_to_dislike_ratio, youtube_likeCount])
                    csvfile.close()
                    firstTimeWrite=False
                else:
                    filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewriter.writerow([revenue, title, initial_budget, truescore, youtube_like_to_dislike_ratio, youtube_likeCount])
                    csvfile.close()
        except:
            print("Error opening the file! Make sure Train.csv is valid and exists.")


        title, initial_budget, revenue, truescore = "", 0, 0, 0 
        youtube_like_to_dislike_ratio, youtube_likeCount, youtube_total_view = 0, 0, 0
        
        stopString = input("\nMining complete. Do the next batch?\nType 'y' to continue collecting data, everything else to quit.")
        if(stopString=='y'):
            pass
        else:
            stop=True


if __name__ == "__main__":    
    program()

    
    


        
