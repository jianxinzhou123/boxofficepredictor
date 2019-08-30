import twitterz
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as analyzer
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk import classify 
from random import shuffle
from nltk import FreqDist
from nltk.corpus import stopwords
import string

positiveReviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    positiveReviews.append(words)

negativeReviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    negativeReviews.append(words)

stopwords_en = stopwords.words('english')

def wordbag(words):
    words_with_no_noises = []
 
    for word in words:
        word = word.lower()
        if word not in stopwords_en and word not in string.punctuation:
            words_with_no_noises.append(word)
    
    words_dictionary = dict([word, True] for word in words_with_no_noises)
    
    return words_dictionary

positiveReviews_set = []
for words in positiveReviews:
    positiveReviews_set.append((wordbag(words), 'pos'))

negativeReviews_set = []
for words in negativeReviews:
    negativeReviews_set.append((wordbag(words), 'neg'))

shuffle(negativeReviews_set)
shuffle(positiveReviews_set)

test_set = positiveReviews_set[:400] + negativeReviews_set[:400]
train_set = positiveReviews_set[400:] + negativeReviews_set[400:]

classifier = NaiveBayesClassifier.train(train_set)



def getData(domain, count):
    datas = twitterz.fetchTweets(domain, count)
    return datas


def sentimentScoreEval(datas):

    total_data_count = 0
    total_sentiment_score = 0.0 #intensity analyzed data
    total_sentiment_average_score = 0.0
    
    for comment in datas:
        comment_tokens = word_tokenize(comment)
        comment_feature_set = wordbag(comment_tokens)
        confidence = classifier.prob_classify(comment_feature_set)
        
        if(confidence.max()=='pos'):
            total_sentiment_score += 1
            total_data_count+=1
        else:
            total_sentiment_score -= 1
            total_data_count+=1

    total_sentiment_average_score = total_sentiment_score / total_data_count

    return total_sentiment_average_score 
            
def sentimentIntensityScoreEval(datas):

    total_data_count =0
    total_sentiment_intensity_score = 0.0 #intensity analyzed data
    total_sentiment_intensity_average_score = 0.0
    reception = ""

    hal = analyzer()
    for comment in datas:
        #print(comment)
        ps = hal.polarity_scores(comment)
        #for k in sorted(ps):
        #    print('\t{}: {:>1.4}'.format(k, ps[k]), end='  ')
        total_data_count += 1
        total_sentiment_intensity_score += ps['compound']
        
    total_sentiment_intensity_average_score = total_sentiment_intensity_score / total_data_count

    return total_sentiment_intensity_average_score


def trueScoreEval(datas):
    sentimentScore = sentimentScoreEval(datas)
    sentimentIntensityScore = sentimentIntensityScoreEval(datas) 
    true_score = (sentimentScore + sentimentIntensityScore) /2
    
    if(true_score > 0.1):
        reception = "positive"
    elif(true_score <= 0.1 and true_score >= -0.1):
        reception = "neutral"
    else:
        reception = "negative"
        
    print("-The average sentiment score, based on the data gathered, is: " + str(float(sentimentScore)) + ".")
    print("-The average sentiment intensity score, based on the data gathered, is: " + str(float(sentimentIntensityScore)) + ".")
    print("-Overall, the data sediment is: " + reception + " with a TwitterSense true score of " + str(float(true_score)) + ".")

    return true_score



if __name__ == '__main__':
    print("Please run the correct program! It's called run_predictor.py!")

    




