#!/usr/bin/env python
"""Generate a set of time shifting models from the given period of years. Each
one of the models spans a given number of years.

Usage:
  runGenerateModels.py --start=<startYear> --end=<endYear> --n=<years> --out=<dir> [--step=<years>] --index=<index> [--lang=<language>] --field=<field> [--shift=<years>] [--mc=<minCount>] [--dim=<dimensions>] [--mv=<maxVocabSize>]

Options:
  --start <startYear>   First year in the generated models
  --end <endYear>   Last year in the generated models
  --n <years>   Number of years per model
  --out <dir>   Directory where models will be writen to
  --shift <years>   Step between start year of generated models [default: 1]
  --index <index>   Which index to use for generating models
  --lang <language> Which language stopword list to use [default: english]
  --field <field>   Which field in the Elasticsearch index is used for extracting sentences
  --mc <minCount> Minimum frequency of a token to be considered in the vocabulary [default: 50]
  --dim <dimensions> The number of dimensions of the resulting word vectors [default: 128]
  --mv <maxVocabSize> Maximum size of vocabulary, useful to set when training toy models [default: None]
"""
import csv
from os.path import join
import os
import pickle

from docopt import docopt
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from collect_sentences import DataCollector
from util import check_path

import logging
logging.basicConfig(filename='models.log', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %H:%M:%S', 
    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

MIN_COUNT = 50
N_DIMS = 128

def generate_models(
        start_year,
        end_year,
        years_in_model,
        model_folder,
        index,
        field,
        shift_years=1,
        language='english',
        min_count=MIN_COUNT,
        vector_size=N_DIMS,
        max_vocab_size=None):
    """Generate time shifting w2v models on the given time range (start_year - end_year).
    Each model contains the specified number of years (years_in_model). The start
    year of each new model is set to be shift_years after the previous model.
    Resulting models are saved on model_folder.
    Generates the following files:
    - a full model file ('*full.model')
    - the word vectors (gensim KeyedVectors) of the full model ('*full.w2v')
    - the analyzer used for preprocessing raw data ('*full_analyzer.pkl)
    - the vocabulary of the full model ('*full.vocab')
    - for each time bin:
        - its word vectors (gensim KeyedVectors) ('*start-end.w2v')
        - the number of tokens (after preprocessing such as stopword removal)
        - the number of terms (i.e., distinct words)
        The statistics are saved to the model folder as a .csv
    """
    check_path(model_folder)
    stopword_list = stopwords.words(language)
    cv = CountVectorizer(stop_words=stopword_list)
    analyzer = cv.build_analyzer()
    sentences = DataCollector(index, start_year, end_year, analyzer, field, model_folder)
    full_model_name = '{}_{}_{}_full'.format(index, start_year, end_year)
    full_model_file =  '{}.model'.format(full_model_name)
    if not os.path.exists(join(model_folder, full_model_file)):
        model = Word2Vec(min_count=int(min_count), vector_size=int(vector_size), max_vocab_size=int(max_vocab_size) if max_vocab_size else None)
        model.build_vocab(sentences)
        # save the analyzer
        vectorizer_name = join(model_folder, '{}_analyzer.pkl'.format(
            full_model_name))
        with open(vectorizer_name, 'wb') as f:
            pickle.dump(analyzer, f)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(join(model_folder, full_model_file))
        model.init_sims(replace=True)
        model.wv.save_word2vec_format(
            join(model_folder, '{}.w2v'.format(full_model_name)),
            binary=True
        )
        vocab = list(model.wv.key_to_index.keys())
        vocab_name = join(model_folder, '{}_vocab.pkl'.format(
            full_model_name))
        with open(vocab_name, 'wb') as f:
            pickle.dump(vocab, f)

    stats = []

    for year in range(start_year, end_year - years_in_model + 1, shift_years):
        start = year
        end = year + years_in_model
        model = Word2Vec.load(join(model_folder, full_model_file))
        logger.info('Training model for year {}'.format(year))
        model_name = '{}_{}_{}.w2v'.format(index, start, end)
        vocab_name = '{}_{}_{}_vocab.pkl'.format(index, start, end)
        logger.info('Building model: '+ model_name)
        sentences = DataCollector(index, start, end, analyzer, field, model_folder)
        cv = CountVectorizer(analyzer=lambda x: x)
        doc_term = cv.fit_transform(sentences)
        cv_vocab = cv.get_feature_names()
        stats.append({
            'time': '{}-{}'.format(start, end),
            'n_tokens': doc_term.sum(),
            'n_terms': len(vocab)})
        
        model.train(sentences, total_examples=len(list(sentences)), epochs=model.epochs)
        vocab = list(set(model.wv.key_to_index.keys()).intersection(set(cv_vocab)))
        with open(join(model_folder, vocab_name), 'wb') as vocab_file:
            pickle.dump(vocab, vocab_file)
        # init_sims precomputes the L2 norm, model cannot be trained further after this step
        model.init_sims(replace=True)
        
        model.wv.save_word2vec_format(join(model_folder, model_name), binary=True)
        
    with open(join(model_folder, '{}_stats.csv'.format(full_model_name)), 'w+') as f:
        writer = csv.DictWriter(f, fieldnames=('time', 'n_tokens', 'n_terms'))
        writer.writeheader()
        writer.writerows(stats)
    

if __name__ == '__main__':
    args = docopt(__doc__)
    years_in_model = int(args['--n'])
    shift_years = int(args['--shift'])
    model_folder = args['--out']
    start_year = int(args['--start'])
    end_year = int(args['--end'])
    index = args['--index']
    language = args['--lang']
    field = args['--field']
    min_count = int(args['--mc'])
    vector_size = int(args['--dim'])
    max_vocab_size = int(args['--mv'])

    generate_models(
        start_year,
        end_year,
        years_in_model,
        model_folder,
        index,
        field,
        shift_years,
        language,
        min_count,
        vector_size,
        max_vocab_size
    )
