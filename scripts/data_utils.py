import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def get_balanced_data(num, ignored_categories=[]):
    df = pd.read_csv('../data/merged_data.csv', header=None)
    category_types = df[1].unique()

    texts = []
    categories = []
    sources = []

    for category_type in category_types:
        
        if category_type in ignored_categories:
            continue 
        
        relevant = df[df[1] == category_type]
        
        sample_size = num
        if len(list(relevant[0])) < sample_size:
            sample_size = len(list(relevant[0]))
        
        sampled = relevant.sample(n=sample_size, random_state=1)
        texts += list(sampled[0])
        categories += list(sampled[1])
        sources += list(sampled[2])

    return texts, categories, sources


def visualize_data():
    df = pd.read_csv('../data/merged_data.csv', header=None)
    category_types = df[1].unique()

    data = Counter(df[1])

    values = list(data.values())
    labels = list(data.keys())

    indices = list(np.argsort(values))

    values = [values[i] for i in indices]
    labels = [labels[i] for i in indices]

    #values = values.reverse()
    #labels = labels.reverse()

    fig = plt.figure(figsize=(10,5))
    font = {'size'   : 20, 'weight': 'bold'}
    plt.rc('font', **font)
    plt.barh(labels, values, log=True, align="center")
    plt.xticks([10, 100, 1000, 10000])
    plt.xlabel('Number of documents', fontdict={'size'   : 20, 'weight': 'bold'})
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_data()