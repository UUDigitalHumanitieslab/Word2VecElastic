[![DOI](https://zenodo.org/badge/171899269.svg)](https://zenodo.org/badge/latestdoi/171899269)
# Word2VecElastic
This repository includes utility functions to build diachronic Word2Vec models in gensim, using an Elasticsearch index to collect the data, and SpaCy and NLTK to preprocess it.

The data is read in year batches from Elasticsearch and preprocessed. Every year's preprocessed data is saved to hard disk (as a pickled list of lists of words), so that for multiple passes (e.g., one to build the vocabulary, one to train the model), the data is available more readily.

For the whole time period, a full model will be generated, which will be used as pre-training data for the individual models. Alternatively, independent models can be trained by setting the `-in` flag (see #Usage)

# Prerequisites
The code was tested in Python 3.8. Create a virtualenv (`python -m venv your_env_name`), activate it (`source your_env_name/bin/activate`) and the run
```
pip install -r requirements.txt
```
## NLTK stopwords
Download the NLTK stopword list as follows: with activated environment, log into the Python shell. Then run
```
import nltk
nltk.download('stopwords')
```

## SpaCy language models
With activated environment, download the SpaCy language models required for preprocessing as follows:
```
python -m spacy download en_core_web_sm
```
See (the SpaCy documentation)[https://spacy.io/usage/models].

# Usage
To train models, with activated environment, use the command
```
python generate_models.py -i your-index-name -s 1960 -e 2000 -md /path/to/output/models
```
Meaning of the flags:
- i: name of the index
- s: start year of training
- e: end year of training
- md: set the output directory where models will be written
Optional flags:
- f: field from which to read the training data (default: 'content')
- n: number of years per model (default: 10)
- sh: shift between models (default: 5)
- sd: path to output preprocessed training data (default: 'source_data')
- l: language of the training data (default: 'english')
- mc: minimum count of a word in the training data to be included in the word models' vocabulary (default: 100)
- vs: size of the word embedding vectors (default: 100)
- mv: set to integer (e.g., 50000) if the vocabulary should be pruned while training the model, without a value provided, there is no limit on vocabulary size
- ws: window size of the words to be trained (default: 5)
- lem: set this flag if you want the data to be lemmatized
- in: set this flag if you want to train independent models, i.e., models which do not depend on data from other time slices
You can also run
`python generate_models.py -h` to see this documentation.
