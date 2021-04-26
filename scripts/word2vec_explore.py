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
from sklearn.feature_extraction.text import TfidfVectorizer



def tokenize_and_lemmatize(text):
    text = re.sub("[@]\w+", "", text) # remove tags
    text = re.sub("<user>", "", text, flags=re.IGNORECASE) # remove <USER> tags
    text = re.sub("<url>", "", text, flags=re.IGNORECASE) # remove <URL> tags
    # remove punctuation
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    text = re.sub("(http)\w+", "", text) # remove links
    tokens = nltk.word_tokenize(text)
    filtered_tokens = []
    # Filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation).
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    #lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return filtered_tokens

lemmatizer = WordNetLemmatizer()
englishwords = stopwords.words('english')
englishwords.extend(["u","rt","im","em","q","na","wan","’","“","”","…"])
englishwords.extend(tokenize_and_lemmatize(" ".join(englishwords)))

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

#tweets,categories,sources = get_balanced_data(5000, ['spam', 'derailing', 'dominance', 'stereotype', 'benevolent sexism'])
tweets,categories,sources = get_balanced_data(5000)

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
            similarity_matrix[i][j] = -sorted_indices[j]

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
            similarity_matrix[i][j] = -sorted_indices[j]

    print_matrix(similarity_matrix)

    return similarity_matrix

def reorder_matrix(matrix, index, onlyColumns=False):
    matrix = np.array(matrix)
    for i in range(0, len(index)):
        while i != index[i]:
            if not onlyColumns:
                matrix[[i, index[i]],:] = matrix[[index[i], i],:]
            matrix[:,[i, index[i]]] = matrix[:,[index[i], i]]
            
            temp = index[index[i]]
            index[index[i]] = index[i]
            index[i] = temp 

    return matrix

def visualize(labels, similarity_matrix):

    clustering = AffinityPropagation(random_state=5).fit(similarity_matrix)
    print(clustering.cluster_centers_indices_)
    print(clustering.labels_)
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

    sec_order = np.argsort(order)
    labels = [labels[i] for i in sec_order]
    print(order)
    print(labels)
    similarity_matrix = reorder_matrix(similarity_matrix, order)


    #print_most_similar_categories(unique_categories)
    fig, ax = plt.subplots()
    font = {'size'   : 20, 'weight': 'bold'}
    plt.rc('font', **font)
    heatmap = plt.imshow(similarity_matrix)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    fig.tight_layout()
    #plt.colorbar(heatmap)
    plt.show()

def similarity_15x15():

    keywords = ["offensive", "abusive", "cyberbullying", "vulgar", "racist", "homophobic", "profane", "slur", "harrass", "obscene", "threat", "discredit", "hateful", "insult", "hostile"]

    path = os.path.join(base_dir, 'word2vec-google-news-300', "word2vec-google-news-300.gz")
    model = KeyedVectors.load_word2vec_format(path, binary=True)

    similarity_matrix = [[0 for i in range(0,len(keywords))] for j in range(0,len(keywords))]

    for i in range(0, len(keywords)):
        for j in range(0, len(keywords)):
            if keywords[i] in model and keywords[j] in model:
                similarity_matrix[i][j] = model.similarity(keywords[i], keywords[j])
            else:
                print(keywords[i])
    visualize(keywords, similarity_matrix)

    return similarity_matrix

#similarity_matrix = similarity_15x15()#build_symetric_matrix(unique_categories) #build_similarity_matrix(unique_categories) #


def similarity_19x15():
    keywords = ["offensive", "abusive", "cyberbullying", "vulgar", "racist", "homophobic", "profane", "slur", "harrassment", "obscene", "threat", "discredit", "hateful", "insult", "hostile"]

    path = os.path.join(base_dir, 'word2vec-google-news-300', "word2vec-google-news-300.gz")
    model = KeyedVectors.load_word2vec_format(path, binary=True)


    # zdruzis tweete z isto kategorijo v isti dokument -> dobimo 19 dokumentov
    uniquecategories = np.unique(categories)
    tweetsNP = np.array(tweets)
    tweetsByCategory = []
    for i in range(len(uniquecategories)):
        indices = np.where(np.array(categories) == uniquecategories[i])
        tweetsByCategory.append(" ".join(list(tweetsNP[indices])))
    uniquecategories = np.array(["benevolent" if c == "benevolent sexism" else "hostile" if c == 'hostile sexism' else "harassment" if c == 'sexual_harassment' else c for c in uniquecategories])


    # zgradim tf-idf
    tfidf_vec3 = TfidfVectorizer(max_df=1.0,
                        max_features=10000,
                        min_df=3, 
                        stop_words=englishwords, 
                        tokenizer=tokenize_and_lemmatize, 
                        ngram_range=(1,1))

    tfidf_mat3 = tfidf_vec3.fit_transform(tweetsByCategory)

    documentVectors = np.zeros((300,19))
    npFeatures = np.array(tfidf_vec3.get_feature_names())
    for i in range(19):
        row = np.array(tfidf_mat3.getrow(i).todense()).flatten()
        indices = row.argsort()[::-1][:30]
        print(uniquecategories[i], npFeatures[indices])
        for ix in indices:
            if npFeatures[ix] in model:
                documentVectors[:,i] += model[npFeatures[ix]] * row[ix]

    similarity_matrix = [[0 for i in range(0,len(keywords))] for j in range(0,19)]

    for i in range(0, 19):
        for j in range(0, len(keywords)):
            if keywords[j] in model:
                similarity_matrix[i][j] = np.dot(model[keywords[j]], documentVectors[:,i]) / (np.linalg.norm(model[keywords[j]]) * np.linalg.norm(documentVectors[:,i]))
            else:
                print(keywords[i])
    
    print("Most similar words:")
    topN = []
    for i in range(0,15):
        print(keywords[i], np.array(unique_categories)[np.argsort(np.transpose(np.transpose(similarity_matrix)[i]))[-3:]])


    clustering = AffinityPropagation(random_state=5).fit(np.transpose(np.array(similarity_matrix)))
        
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

    sec_order = np.argsort(order)

    print(order)

    similarity_matrix = reorder_matrix(similarity_matrix, order, onlyColumns=True)

    fig, ax = plt.subplots()
    heatmap = plt.imshow(similarity_matrix)
    ax.set_xticks(np.arange(len(keywords)))
    ax.set_xticklabels([keywords[i] for i in sec_order])
    ax.set_yticks(np.arange(len(uniquecategories)))
    ax.set_yticklabels(uniquecategories)
    plt.ylabel('Data categories', fontdict={'size': 15})
    plt.xlabel('Terms', fontdict={'size': 15})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    fig.tight_layout()
    #plt.colorbar(heatmap)
    plt.show()

similarity_19x15()
#similarity_15x15()
"""
path = os.path.join(base_dir, 'word2vec-google-news-300', "word2vec-google-news-300.gz")
model = KeyedVectors.load_word2vec_format(path, binary=True)
w,e = get_similar_words(model, ["abusive"], 50)
print(w)
w,e = get_similar_words(model, ["threat"], 50)
print(w)
w,e = get_similar_words(model, ["insult"], 50)
print(w)
"""