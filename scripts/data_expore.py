from data_utils import get_balanced_data
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
import re
import string
import nltk


tweets, categories, sources = get_balanced_data(5000)

unique_categories = list(np.unique(categories))

lengths = [0] * len(unique_categories)
numbers = [0] * len(unique_categories)


for j in range(0, len(unique_categories)):
    category = unique_categories[j]
    for i in range(len(tweets)):
        if category == categories[i]:
            lengths[j] += len(tweets[i])
            numbers[j] += 1

average_lengths = [lengths[i]/numbers[i] for i in range(0, len(lengths))]

for i in range(0, len(average_lengths)):
    print(unique_categories[i], int(average_lengths[i]))

lemmatizer = WordNetLemmatizer()
englishwords = stopwords.words('english')
englishwords.extend(["u","rt","im","em","q","na","wan","’","“","”","…"])

def tokenize_and_lemmatize(text):
    text = re.sub("[@]\w+", "", text) # remove tags
    text = text.lower()
    # remove punctuation
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    text = re.sub("(http)\w+", "", text) # remove links
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in englishwords]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


processed_list = []

for j in range(0, len(unique_categories)):
    current_list = []
    category = unique_categories[j]
    for i in range(len(tweets)):
        if category == categories[i]:
            current_list += tokenize_and_lemmatize(tweets[i])
    processed_list.append(current_list)


for i in range(0, len(processed_list)):
    print(unique_categories[i], Counter(processed_list[i]).most_common(10))
