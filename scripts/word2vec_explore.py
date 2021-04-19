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
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

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

def print_matrix(a):
    for row in a:
        print(row)

def build_symetric_matrix(unique_categories):
    word_clusters, embedding_clusters = get_word2vec_words(unique_categories)
    vector = merge_vectors(word_clusters)

    vectors = []

    for i in range(0, len(unique_categories)):
        occurencies = [0] * len(vector)
        for j in range(0, len(word_clusters[i])):
            for n in range(0, len(vector)):
                if vector[n] == word_clusters[i][j]:
                    occurencies[n] += 1
        vectors.append(occurencies)

    similarity_matrix = [[0 for i in range(0,len(vectors))] for j in range(0,len(vectors))]

    for i in range(0, len(vectors)):

        values = [0]*len(vectors)

        for j in range(0, len(vectors)):
            if i == j:
                continue
            values[j] = compare_vectors(vectors[i], vectors[j])

        sorted_indices = values
        for j in range(0, len(vectors)):
            if i == j:
                continue
            similarity_matrix[i][j] = sorted_indices[j]

    print_matrix(similarity_matrix)

    return similarity_matrix

def build_similarity_matrix(unique_categories):
    word_clusters, embedding_clusters = get_word2vec_words(unique_categories)
    vector = merge_vectors(word_clusters)

    vectors = []
    for i in range(0, len(word_clusters)):
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

    similarity_matrix = [[0 for i in range(0,len(vectors))] for j in range(0,len(vectors))]

    for i in range(0, len(vectors)):

        values = [0]*len(vectors)

        for j in range(0, len(vectors)):
            if i == j:
                continue
            values[j] = compare_vectors(vectors[i], vectors[j])

        sorted_indices = values #np.argsort(np.argsort(values))
        for j in range(0, len(vectors)):
            if i == j:
                continue
            similarity_matrix[i][j] = sorted_indices[j]

    print_matrix(similarity_matrix)

    return similarity_matrix

def reorder_matrix(matrix, index):
    matrix = np.array(matrix)
    for i in range(0, len(index)):
        while i != index[i]:
            matrix[[i, index[i]],:] = matrix[[index[i], i],:]
            matrix[:,[i, index[i]]] = matrix[:,[index[i], i]]
            
            temp = index[index[i]]
            index[index[i]] = index[i]
            index[i] = temp 

    return matrix

similarity_matrix = build_similarity_matrix(unique_categories) #build_symetric_matrix(unique_categories)
"""
similarity_matrix = [[0, 3, 13, 11, 8, 9, 2, 10, 5, 12, 4, 7, 6],
[2, 0, 3, 5, 10, 9, 12, 8, 13, 1, 4, 7, 6],
[3, 2, 0, 5, 9, 10, 12, 1, 8, 13, 4, 7, 6],
[4, 1, 8, 0, 7, 12, 6, 3, 9, 11, 5, 2, 10],
[5, 2, 9, 11, 0, 3, 12, 8, 1, 13, 4, 7, 6],
[6, 7, 4, 12, 8, 0, 1, 3, 9, 11, 2, 10, 5],
[7, 6, 4, 13, 12, 8, 0, 9, 3, 10, 11, 2, 5],
[8, 3, 12, 11, 1, 2, 10, 0, 5, 9, 4, 7, 6],
[9, 11, 5, 2, 10, 3, 12, 1, 0, 8, 4, 7, 6],
[10, 11, 2, 5, 9, 3, 12, 8, 13, 0, 4, 7, 6],
[11, 2, 3, 10, 9, 5, 12, 8, 1, 13, 0, 7, 6],
[12, 11, 3, 2, 10, 5, 8, 9, 13, 1, 4, 0, 6],
[13, 3, 1, 11, 10, 8, 2, 12, 9, 5, 4, 7, 0]]
"""



clustering = AffinityPropagation(random_state=5).fit(similarity_matrix)
print(clustering.cluster_centers_indices_)
print(clustering.labels_)
labels = []
freqeuncy = {}
for i in clustering.labels_:
    if clustering.cluster_centers_indices_[i] not in freqeuncy:
        freqeuncy[clustering.cluster_centers_indices_[i]] = 1
    else:
        freqeuncy[clustering.cluster_centers_indices_[i]] += 1

keys = sorted(freqeuncy.keys())

cumsum = [0]*len(keys)

for i in range(1, len(keys)):
    for j in range(0,i):
        cumsum[i] += freqeuncy[keys[j]]

order = []

for i in range(0, len(clustering.labels_)):
    order.append(cumsum[clustering.labels_[i]])
    cumsum[clustering.labels_[i]] += 1

similarity_matrix = reorder_matrix(similarity_matrix, order)

print(order)

#print_most_similar_categories(unique_categories)
print([unique_categories[i] for i in order])
fig, ax = plt.subplots()
heatmap = plt.imshow(similarity_matrix)
ax.set_yticks(np.arange(len(unique_categories)))
ax.set_yticklabels([unique_categories[i] for i in order])
ax.set_xticks(np.arange(len(unique_categories)))
ax.set_xticklabels([unique_categories[i] for i in order])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.colorbar(heatmap)
plt.show()

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



