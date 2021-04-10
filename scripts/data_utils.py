import pandas as pd

def get_balanced_data(num):
    df = pd.read_csv('../data/merged_data.csv', header=None)
    category_types = df[1].unique()

    texts = []
    categories = []
    sources = []

    for category_type in category_types:
        relevant = df[df[1] == category_type]
        
        sample_size = num
        if len(list(relevant[0])) < sample_size:
            sample_size = len(list(relevant[0]))

        sampled = relevant.sample(n=sample_size, random_state=1)
        texts += list(sampled[0])
        categories += list(sampled[1])
        sources += list(sampled[2])

    return texts, categories, sources

if __name__ == '__main__':
    a,b,c = get_balanced_data(500)

    print(a[350:360])
    print(b[350:360])
    print(c[350:360])