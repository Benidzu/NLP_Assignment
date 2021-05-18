import pandas as pd
from collections import Counter
from start import conan, sexists, mmhs150k, iberEval, kaggle, founta, hasoc, cyberbullying, vulgar, davidson

def merge_conan(merged_file):
    texts, categories = conan()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'conan\n')

def merge_sexists(merged_file):
    texts, categories = sexists()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        if texts[i] == "":
            continue
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'jha\n')

def merge_iberEval(merged_file):
    texts, categories = iberEval()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'iberEval\n')

def merge_kaggle(merged_file):
    texts, categories = kaggle()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'kaggle\n')

def merge_founta(merged_file):
    texts, categories = founta()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'founta\n')

def merge_hasoc(merged_file):
    texts, categories = hasoc()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'hasoc\n')

def merge_mmhs150k(merged_file):
    texts, categories = mmhs150k()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'mmhs150k\n')

def merge_cyberbullying(merged_file):
    texts, categories = cyberbullying()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'reynolds\n')

def merge_vulgar(merged_file):
    texts, categories = vulgar()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'cachola\n')

def merge_davidson(merged_file):
    texts, categories = davidson()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        texts[i] = texts[i].replace("\n", " ")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'davidson\n')

def rewrite_data():
    """
    This function combines different datasets in one merged .csv file that was used in our assignment. Due to the data size those original dataset files are not included in the GitHub repository. Therefore, this function should not be run, unless one has all data saved locally.
    """
    file_path = '../data/merged_data.csv'
    merged_file = open(file_path, 'w', encoding='utf-8')
    merge_cyberbullying(merged_file)
    merge_davidson(merged_file)
    merge_founta(merged_file)
    merge_hasoc(merged_file)
    merge_iberEval(merged_file)
    merge_kaggle(merged_file)
    merge_mmhs150k(merged_file)
    merge_sexists(merged_file)
    merge_vulgar(merged_file)
    merged_file.close()

def inspect_merged_data():
    print('Number of texts per category:')
    dfTrain = pd.read_csv("../data/merged_data.csv", sep=",", header=None)
    sez = Counter(list(dfTrain[1]))
    for s in sez:
        print(s, sez[s])

#rewrite_data()
inspect_merged_data()

