# Offensive language exploratory analysis

## Repository organization

### Data

``data/merged_data.csv`` contains merged data gathered from our data sources, which is used for all our analysis. We gathered and organized this data mostly automatically using some scripts as described below. For the purpose of evaluation of our code, this data we prepared and included suffices and no additional data gathering/preprocessing is required. All the gathered data is publicly accessible, we list the sources in the following table, as well as in the report.

| Dataset | source  | 
| :---:   | :-: | 
| Davidson | https://github.com/t-davidson/hate-speech-and-offensive-language |
| Reynolds | https://www.chatcoder.com/data.html |
| Founta | https://github.com/ENCASEH2020/hatespeech-twitter |
| Mandl | https://hasocfire.github.io/hasoc/2019/dataset.html |
| Cachola | https://github.com/ericholgate/VulgarFunctionsTwitter |
| Kaggle | https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data |
| Fersini | https://amiibereval2018.wordpress.com/important-dates/data/ |
| Gomez | https://gombru.github.io/2019/10/09/MMHS/ |
| Jha | https://github.com/AkshitaJha/NLP_CSS_2017 |

### Report

``report/`` includes the PDF report file, associated with this repository.

### Source code

Our main source code is contained in ``scripts/`` and ``notebooks/``

``notebooks/`` contains the main jupyter notebooks used for our analysis.
* ``generalExploration.ipynb`` contains the majority of our exploration on merged data.
* ``iberEvalExploration.ipynb`` contains some of our early analysis on the IberEval dataset.

Additionally, the Colab notebook which can be accessed [here](https://colab.research.google.com/drive/1xesfr4uBJJs11hAhujwsS79hfOZLYk2A?usp=sharing) and requires no additional setup, contains our analysis using BERT and ELMo.

``scripts/`` contains python scripts, mostly connected to data preparation. Specifically:
* ``data_gatherer.py`` is an script used for retrieval of tweets. It's was modified specifically for every dataset in order to accomodate to different dataset structures.
* ``twitter.py`` contains code for retrieving tweets. API key required.
* ``start.py`` contains code for parsing different datasets.
* ``data_merger.py`` is a script we used to merge all the data to ``merged_data.csv``.
* ``data_utils.py`` contains the utilities for sampling the merged data as well as a simple frequency plot.
* ``data_explore.py`` is script used for exploring our dataset.
* ``word2vec_explore.py`` contains our analysis of offensive terms based on word2vec similarities with data.

It should be again noted that scripts ``data_gatherer.py``, ``twitter.py``, ``start.py`` and ``data_merger.py`` were needed in order to obtain our dataset present in ``data/merged_data.csv``. But these scripts use data from different datasets which are not present in this GitHub repository, since files are too big. For the original data sources refer to the Data section above.  

## Environment setup

The below instructions were succesfully tested on Windows 10 64-bit, using Anaconda.

1) Create a new conda environment based on requirements from ``env.yml``: 
```
conda env create -f env.yml

conda activate nlp_env
```
2) Using VSCode, or any other IDE, or the installed jupyter library and your web browser, access the notebooks from ``notebooks/``. For example using the jupyter library, launch the server and follow the instructions from the terminal via:
```
jupyter notebook
```
Don't forget to trust the notebooks and choose the correct kernel when attempting to run them.

3) For the scripts ``data_utils.py``, ``data_explore.py`` and ``word2vec_explore.py``, you can run them simply from the command line:
```
cd scripts

python data_utils.py

python data_explore.py

python word2vec_explore.py
```

4) The additional colab notebook can be accessed [here](https://colab.research.google.com/drive/1xesfr4uBJJs11hAhujwsS79hfOZLYk2A?usp=sharing) 

A pre-trained word2vec model is required for evaluation of some parts of code. It will be downloaded automatically if not present.