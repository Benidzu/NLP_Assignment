import gensim
import gensim.downloader
from gensim.models import KeyedVectors
import os
from gensim.models import KeyedVectors
from gensim.downloader import base_dir
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import json
import nltk
import string
import re
import sys
sys.path.append("../")
from scripts.data_utils import get_balanced_data
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
englishwords = stopwords.words('english')
englishwords.extend(["u","rt","im","em","q","na","wan","’","“","”","…"])

def tokenize_and_lemmatize(text):
    text = re.sub("[@]\w+", "", text) # remove tags
    # remove punctuation
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    text = re.sub("(http)\w+", "", text) # remove links
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in englishwords]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def get_similar_words(model, categories, n=30):
    embedding_clusters = []
    word_clusters = []
    for category in categories:
        embeddings = []
        words = []
        try:
            most_similar_words = model.most_similar(category, topn=n)

            for similar_word, _ in most_similar_words:
                words.append(similar_word)
                embeddings.append(model[similar_word])

            if len(words) < n or len(embeddings) < n:
                print("ERROR")
        except KeyError:
            print(category, "is not in the vocabulary of word2vec")
        words = words[:n]
        embeddings = embeddings[:n]
        
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
        
    return (word_clusters, embedding_clusters)

def get_indexes(categories, category):
    start_index = -1
    end_index = -1
    for i in range(len(categories)):
        if categories[i] == category:
            if start_index == -1:
                start_index = i
            end_index = i
        if end_index != -1 and categories[i] != category:
            return start_index, end_index

def get_word2vec_words(unique_categories):
    path = os.path.join(base_dir, 'word2vec-google-news-300', "word2vec-google-news-300.gz")
    model = KeyedVectors.load_word2vec_format(path, binary=True, limit=100000)

    return get_similar_words(model, unique_categories)

def train_word2vec_model(name, tweets):
    model = Word2Vec(sentences=tweets)
    model.train(tweets, total_examples=len(tweets), epochs=20)
    model.save(name)

def load_saved_model(name):
    return Word2Vec.load(name)

tweets,categories,sources = get_balanced_data(500)

for i in range(0,len(tweets)):
    tweets[i] = " ".join(tokenize_and_lemmatize(tweets[i]))

unique_categories = list(np.unique(categories))


start, end = get_indexes(categories, "racist")
train_word2vec_model('racist', tweets[start:end+1])
model = load_saved_model('racist')

google_keywords = ['racists', 'racism', 'anti_Semitic', 'bigoted', 'homophobic', 'Racist', 'hateful', 'racially_motivated', 'sexist', 'racial', 'derogatory', 'xenophobic', 'racially_charged', 'racial_slurs', 'slurs', 'insulting', 'bigots', 'vile', 'bigot', 'disrespectful', 'bigotry', 'slur', 'racial_slur', 'offended', 'slanderous', 'demeaning', 'Racism', 'racially', 'racial_hatred', 'despicable']

print(tweets[start:start+5])

for word in google_keywords:
    try:
        print(model.wv.most_similar(positive=word))
    except KeyError:
        pass



"""
word_clusters, embedding_clusters = get_word2vec_words(unique_categories)
for i in range(0, len(word_clusters)):
    print(unique_categories[i], word_clusters[i])
"""