import pandas as pd
import json
import nltk
import string
import re
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

"""
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("reuters")
nltk.download("gutenberg")
nltk.download("wordnet")
nltk.download("tagsets")
nltk.download('punkt')
nltk.download("stopwords")
"""

def conan():

    conan_file = open("../data/CONAN.json", encoding="utf8")

    data = json.load(conan_file)

    texts = []
    categories = []
    categories_count = {}

    for tweet in data["conan"]:
        texts.append(tweet['hateSpeech'])
        categories.append(tweet['hsSubType'])
        if tweet['hsSubType'] in categories_count.keys():
            categories_count[tweet['hsSubType']] += 1
        else:
            categories_count[tweet['hsSubType']] = 1

    text_no_duplicates = []
    categories_no_duplicates = []

    for i in range(0, len(texts)):
        if texts[i] not in text_no_duplicates:
            text_no_duplicates.append(texts[i])
            categories_no_duplicates.append(categories[i])

    texts = text_no_duplicates
    categories = categories_no_duplicates



    return texts[:406], categories[:406] # vecina tweetov po 406 je v italijanscini

def sexists():

    benevolent_file = open("../data/sexists/benevolent_texts.csv", encoding="utf8")
    hostile_file = open("../data/sexists/hostile_texts.csv", encoding="utf8")
    
    benevolent_texts = []
    hostile_texts = []

    for line in benevolent_file:
        text = line[19:].strip()
        if text != "ERROR: no tweet" and text not in benevolent_texts:
            benevolent_texts.append(text)

    
    for line in hostile_file:
        text = line[19:].strip()
        if text != "ERROR: no tweet" and text not in hostile_texts:
            hostile_texts.append(text)

    texts = benevolent_texts + hostile_texts
    categories = ["benevolent sexism"]*len(benevolent_texts) + ["hostile sexism"]*len(hostile_texts)
    
    return texts,categories

def mmhs150k():

    
    with open('../data/MMHS150K/MMHS150K_GT.json','r') as my_file:
        tweets = json.load(my_file)

    texts = []
    categories = []
    categories_count = {}

    for tweetId in tweets:
        texts.append(tweets[tweetId]['tweet_text'])
        categories.append(tweets[tweetId]['labels_str'])

    text_no_duplicates = []
    categories_no_duplicates = []

    for i in range(0, len(texts)):
        if texts[i] not in text_no_duplicates:
            text_no_duplicates.append(texts[i])
            categories_no_duplicates.append(categories[i])

    texts = text_no_duplicates
    categories = categories_no_duplicates

    return texts, categories

def iberEval():
    dfTrain = pd.read_csv("../data/iberEval/en_AMI_TrainingSet.csv", sep=";")
    mysog_tweets = dfTrain[dfTrain["misogyny_category"] != "0"]
    return list(mysog_tweets["tweet"]), list(mysog_tweets["misogyny_category"])

if __name__ == '__main__':
    texts, categories = sexists()

    stopwords = stopwords.words('english')
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    for i in range(0, len(texts)):
        text = texts[i]
        text = re.sub("[@]\w+", "", text) # odstranimo tag-anje drugih ljudi v tweetih
        table = text.maketrans({key: None for key in string.punctuation})
        text = text.translate(table)      
        text = re.sub("(http)\w+", "", text) # odstranimo povezave na spletne strani
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in stopwords]
        generalised = [stemmer.stem(token) for token in tokens]
        texts[i] = generalised

    for i in range(1,50):
        print(i,texts[i])
