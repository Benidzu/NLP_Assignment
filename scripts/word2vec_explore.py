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
    return start_index, end_index

def get_word2vec_words(unique_categories):
    path = os.path.join(base_dir, 'word2vec-google-news-300', "word2vec-google-news-300.gz")
    model = KeyedVectors.load_word2vec_format(path, binary=True, limit=100000)

    return get_similar_words(model, unique_categories)

def train_word2vec_model(name, tweets):
    model = Word2Vec(sentences=tweets, size=300, window=8, min_count=1, sg=1, iter=30)
    model.train(tweets, total_examples=len(tweets), epochs=20)
    model.save(name)

def load_saved_model(name):
    return Word2Vec.load(name)

def merge_vectors(vectors):
    vector = []
    for v in vectors:
        for keyword in v:
            if keyword not in vector:
                vector.append(keyword)
    return vector

def compare_vectors(v1, v2):
    #return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    score = 0
    for i in range(0, len(v1)):
        if (v1[i] == 0 and v2[i] != 0) or (v1[i] != 0 and v2[i] == 0):
            score += 100
        score += abs(v1[i] - v2[i])
    return score

tweets,categories,sources = get_balanced_data(5000, ['spam', 'derailing', 'dominance', 'stereotype'])

for i in range(0,len(tweets)):
    tweets[i] = " ".join(tokenize_and_lemmatize(tweets[i]))

unique_categories = list(np.unique(categories))

"""
start, end = get_indexes(categories, "abusive")
train_word2vec_model('abusive', tweets[start:end+1])
model = load_saved_model('abusive')
"""
"""
word_clusters, embedding_clusters = get_word2vec_words(unique_categories)
for i in range(0, len(word_clusters)):
    print(unique_categories[i], word_clusters[i])
"""

def print_most_similar_categories(unique_categories):

    word_clusters, embedding_clusters = get_word2vec_words(unique_categories)
    vector = merge_vectors(word_clusters)

    vectors = []
    for i in range(0, len(word_clusters)):
        #google_keywords = word_clusters[i]
        google_keywords = vector
        occurencies = [0] * len(google_keywords)
        start, end = get_indexes(categories, unique_categories[i])
        for n in range(0, len(google_keywords)):
            for j in range(start, end+1):
                tweet = tweets[j].split(" ")
                for token in tweet:
                    if token == google_keywords[n]:
                        occurencies[n] += 1
        vectors.append(occurencies)

    for i in range(0, len(vectors)):
        print(unique_categories[i], 'is most similiar to: ', end="")
        min_j = -1
        for j in range(0, len(vectors)):
            if i == j:
                continue
            if min_j == -1:
                min_j = j
                continue
            if compare_vectors(vectors[i], vectors[min_j]) > compare_vectors(vectors[i], vectors[j]):
                min_j = j
        print(unique_categories[min_j])


print_most_similar_categories(unique_categories)

"""
Example input of print_most_similar_categories

abusive is most similiar to: discredit
cyberbullying is most similiar to: sexual_harassment
discredit is most similiar to: cyberbullying
hateful is most similiar to: abusive
homophobic is most similiar to: cyberbullying
insult is most similiar to: obscene
obscene is most similiar to: insult
offensive is most similiar to: discredit
profane is most similiar to: sexual_harassment
racist is most similiar to: sexual_harassment
sexual_harassment is most similiar to: cyberbullying
threat is most similiar to: sexual_harassment
vulgar is most similiar to: discredit
"""



