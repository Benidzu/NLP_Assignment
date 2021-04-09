import pandas as pd
from collections import Counter
from start import conan, sexists, mmhs150k, iberEval, kaggle, founta, hasoc, cyberbullying, vulgar, davidson

file_path = '../data/merged_data.csv'
merged_file = open(file_path, 'a+', encoding='utf-8')

def merge_conan():
    texts, categories = conan()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'conan\n')

def merge_sexists():
    texts, categories = sexists()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'jha\n')

def merge_iberEval():
    texts, categories = iberEval()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'iberEval\n')

def merge_kaggle():
    texts, categories = kaggle()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'kaggle\n')

def merge_founta():
    texts, categories = founta()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'founta\n')

def merge_hasoc():
    texts, categories = hasoc()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'hasoc\n')

def merge_mmhs150k():
    texts, categories = mmhs150k()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'mmhs150k\n')

def merge_cyberbullying():
    texts, categories = cyberbullying()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'reynolds\n')

def merge_vulgar():
    texts, categories = vulgar()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'cachola\n')

def merge_davidson():
    texts, categories = davidson()
    for i in range(0, len(texts)):
        texts[i] = texts[i].replace("\"", "")
        merged_file.write('\"'+texts[i]+'\",'+categories[i]+','+'davidson\n')

def rewrite_data():
    merge_cyberbullying()
    merge_davidson()
    merge_founta()
    merge_hasoc()
    merge_iberEval()
    merge_kaggle()
    merge_mmhs150k()
    merge_sexists()
    merge_vulgar()

def inspect_merged_data():
    dfTrain = pd.read_csv("../data/merged_data.csv", sep=",", header=None)
    sez = Counter(list(dfTrain[1]))
    for s in sez:
        print(s, sez[s])

inspect_merged_data()

merged_file.close()