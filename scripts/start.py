import pandas as pd
import json
import nltk
import string
import re
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET

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
        categories.append(tweet['hsSubType'].split(",")[0])
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
        if tweets[tweetId]['labels_str'].count("Homophobe") == 3:
            texts.append(tweets[tweetId]['tweet_text'])
            categories.append("homophobic")
        elif tweets[tweetId]['labels_str'].count("Racist") == 3:
            texts.append(tweets[tweetId]['tweet_text'])
            categories.append("racist")

    return texts, categories

def iberEval():
    dfTrain = pd.read_csv("../data/iberEval/en_AMI_TrainingSet.csv", sep=";")
    mysog_tweets = dfTrain[dfTrain["misogyny_category"] != "0"]
    return list(mysog_tweets["tweet"]), list(mysog_tweets["misogyny_category"])

def kaggle():
    dfTrain = pd.read_csv("../data/kaggle/train.csv", sep=",")
    obscene_tweets = dfTrain[dfTrain["obscene"] == 1]
    insult_tweets = dfTrain[dfTrain["insult"] == 1]
    threat_tweets = dfTrain[dfTrain["threat"] == 1]

    obscenes = list(obscene_tweets["comment_text"])
    for i in range(0, len(obscenes)):
        obscenes[i] = obscenes[i].strip().replace("\n", " ")
    insults = list(insult_tweets["comment_text"])
    for i in range(0, len(insults)):
        insults[i] = insults[i].strip().replace("\n", " ")
    threats = list(threat_tweets["comment_text"])
    for i in range(0, len(threats)):
        threats[i] = insults[i].strip().replace("\n", " ")

    return (obscenes + insults + threats), (len(obscenes)*["obscene"]+len(insults)*["insult"]+len(threats)*["threat"])

def founta():
    dfTrain = pd.read_csv("../data/founta/hatespeech_text_label_vote.csv", sep="\t", header=None)
    tweets = dfTrain[dfTrain[1] != "normal"]
    return list(tweets[0]), list(tweets[1])

def hasoc():
    dfTrain = pd.read_csv("../data/hasoc/english_dataset.tsv", sep="\t")
    tweets = dfTrain[dfTrain["task_2"] == "PRFN"]
    return list(tweets["text"]), len(list(tweets["text"]))*["profane"]

def cyberbullying():

    texts = []

    xml_file = '../data/cyberbullying/XMLMergedFile.xml'
    tree = ET.parse(xml_file)

    root = tree.getroot()

    for formspingid in root:
        for datafield in formspingid:
            if datafield.tag == "POST":
                text = ""
                confidence = 0
                for post_details in datafield:
                    if post_details.tag == "TEXT":
                        text = post_details.text
                    elif post_details.tag == "LABELDATA":
                        if post_details[0].text == "Yes":
                            confidence += 1
                if confidence == 3:
                    texts.append(text)


    return texts, len(texts)*["cyberbullying"]

def vulgar():
    dfTrain = pd.read_csv("../data/vulgartwitter-master/data/cleaned_data.tsv", sep="\t")
    tweets = dfTrain[dfTrain["(1) Very Negative"] != 0]
    return list(tweets["Tweet"]), len(list(tweets["Tweet"]))*["vulgar"]

def davidson():
    dfTrain = pd.read_csv("../data/davidson/labeled_data.csv")
    tweets = dfTrain[dfTrain["class"] == 1]
    return list(tweets["tweet"]), len(list(tweets["tweet"]))*["offensive"]

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
