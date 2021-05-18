# Offensive language exploratory analysis

## Repository organization

``data/`` includes data gathered from some of our sources. ``data/merged_data.csv`` is the most important file here, containing merged data from different sources, which is used for our analysis.

``report/`` includes the PDF report file, associated with this repository.

Our main source code is contained in ``scripts/`` and ``notebooks/``


``notebooks/`` contains the main jupyter notebooks used for our analysis.
* ``generalExploration.ipynb`` contains the majority of our exploration on merged data.
* ``iberEvalExploration.ipynb`` contains some of our early analysis on the IberEval dataset.
* ``generalAnalysis.ipynb`` contains some more exploration.

Additionally, the Colab notebook which can be accessed [here](https://colab.research.google.com/drive/1xesfr4uBJJs11hAhujwsS79hfOZLYk2A?usp=sharing) and requires no additional setup, contains our analysis using BERT and ELMo.

``scripts/`` contains python scripts, mostly connected to data preparation. Specifically:
* ``data_gatherer.py`` is an script used for retrieval of tweets. It's was modified specifically for every dataset in order to accomodate to different dataset structures.
* ``twitter.py`` contains code for retrieving tweets. API key required.
* ``start.py`` contains code for parsing different datasets.
* ``data_merger.py`` is a script we used to merge all the data to ``merged_data.csv``.
* ``data_utils.py`` contains the utilities for sampling the merged data as well as a simple frequency plot.
* ``data_explore.py`` is script used for exploring out dataset.
* ``word2vec_explore.py`` contains our analysis of offensive terms based on word2vec similarities with data.

It should be noted that scripts ``data_gatherer.py``, ``twitter.py``, ``start.py`` and ``data_merger.py`` were needed in order to obtain our dataset present in ``data/merged_data.csv``. But this scripts use data from different datasets and this data is not present in this GitHub repository, since files are too big.  

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
Don't forget to trust the notebooks and choose the correct kernel.
