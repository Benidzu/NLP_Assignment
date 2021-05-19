# Offensive language exploratory analysis

## Repository organization

``data/merged_data.csv`` contains merged data gathered from our data sources, which is used for all our analysis. We gathered and organized this data mostly automatically using some scripts as described below. For the purpose of evaluation of our code, the data we prepared suffices and no additional data gathering/preprocessing is required. All data is publicly accessible, for sources refer to the paper.

``report/`` includes the PDF report file, associated with this repository.

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

It should be again noted that scripts ``data_gatherer.py``, ``twitter.py``, ``start.py`` and ``data_merger.py`` were needed in order to obtain our dataset present in ``data/merged_data.csv``. But these scripts use data from different datasets and this data is not present in this GitHub repository, since files are too big.  

## Environment setup

The below instructions were succesfully tested on Windows 10 64-bit, using Anaconda.

1) Create a new conda environment based on requirements from ``env.yml``: 
```
conda env create -f env.yml

conda activate nlp_env
```
2) Using VSCode, or any other IDE, or the installed jupyter library directly, access the notebooks, e.g.:
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