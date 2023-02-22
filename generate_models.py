#!/usr/bin/env python
"""Generate a set of time shifting models from the given period of years. Each
one of the models spans a given number of years.
"""
import csv
from os.path import join
import os
import pickle

import click
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer

from collect_sentences import DataCollector
from analyzer import Analyzer
from util import check_path

import logging
logging.basicConfig(filename='models.log', level=logging.WARNING, filemode='a', datefmt='%Y-%m-%d %H:%M:%S', 
    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

MIN_COUNT = 80
N_DIMS = 100
WINDOW_SIZE = 5

@click.command()
@click.option('-i', '--index', help="Elasticsearch index name from which to request the training data", required=True)
@click.option('-s', '--start_year', help="Year from which to start training", type=int, required=True)
@click.option('-e', '--end_year', help="Year until which to continue training", type=int, required=True)
@click.option('-n', '--n_years', help="Number of years each model should span", type=int, default=10)
@click.option('-sh', '--shift_years', help="Shift between models", type=int, default=5)
@click.option('-md', '--model_directory', help="Directory in which the models should be saved", required=True)
@click.option('-sd', '--source_directory', help="Directory in which the source data should be saved", default='source_data')
@click.option('-f', '--field', help="Field from which to extract training data", default='content')
@click.option('-l', '--language', help="Language of the training data", default='english')
@click.option('-lem', '--lemmatize', help="Whether or not to perform lemmatization", default=False, is_flag=True)
@click.option('-mc', '--min_count', help="Minimum count of a given word to be included in a model", type=int, default=MIN_COUNT)
@click.option('-vs', '--vector_size', help="The size of the embedding vectors", type=int, default=N_DIMS)
@click.option('-ws', '--window_size', help="The size of the window considered for embeddings", type=int, default=WINDOW_SIZE)
@click.option('-mv', '--max_vocab_size', help="Limit the size of the vocab, i.e., prune", type=int)
@click.option('-in', '--independent', help="Train models which don't depend on data from other time slices", default=False, is_flag=True)
def generate_models(
        index,
        start_year,
        end_year,
        n_years,
        shift_years,
        model_directory,
        source_directory,
        field,
        language,
        lemmatize,
        min_count,
        vector_size,
        window_size,
        max_vocab_size,
        independent):
    """Generate time shifting w2v models on the given time range (start_year - end_year).
    Each model contains the specified number of years (years_in_model). The start
    year of each new model is set to be shift_years after the previous model.
    Resulting models are saved in the specified output directory.
    Generates the following files:
    - a full model file ('*full.model')
    - the word vectors (gensim KeyedVectors) of the full model ('*full.wv')
    - a file of statistics on each time slice ('*stats.csv'):
        - the number of tokens (after preprocessing such as stopword removal)
        - the number of terms (i.e., distinct words)
    - for each time bin:
        - its word vectors (gensim KeyedVectors) ('*start-end.wv')
    The statistics are saved to the model folder as a .csv
    """
    check_path(model_directory)
    analyzer = Analyzer(language, lemmatize).preprocess
    sentences = DataCollector(index, start_year, end_year, analyzer, field, source_directory)
    full_model_name = '{}_{}_{}_full'.format(index, start_year, end_year)
    full_model_file =  '{}.model'.format(full_model_name)
    if not os.path.exists(join(model_directory, full_model_file)) and not independent:
        # skip this step when training independent models
        model = get_model(
            sentences,
            min_count,
            window_size,
            vector_size,
            max_vocab_size
        )
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(join(model_directory, full_model_file))
        model.wv.save(
            join(model_directory, '{}.wv'.format(full_model_name))
        )

    stats = []
    for year in range(start_year, end_year - n_years + 1, shift_years):
        start = year
        end = year + n_years
        model_name = '{}_{}_{}.wv'.format(index, start, end)
        logger.info('Building model: '+ model_name)
        sentences = DataCollector(index, start, end, analyzer, field, source_directory)
        if independent:
            model = get_model(
                sentences,
                min_count,
                window_size,
                vector_size,
                max_vocab_size
            )
        else:
            model = Word2Vec.load(join(model_directory, full_model_file))
        output1, n_tokens = model.train(sentences, total_examples=len(list(sentences)), epochs=model.epochs)
        saved_vectors, n_terms, n_tokens = get_vectors_and_stats(
            model, sentences, n_tokens, independent
        )
        stats.append({
            'time': '{}-{}'.format(start, end),
            'n_tokens': n_tokens,
            'n_terms': n_terms})    
        saved_vectors.save(join(model_directory, model_name))
        
    with open(join(model_directory, '{}_stats.csv'.format(full_model_name)), 'w+') as f:
        writer = csv.DictWriter(f, fieldnames=('time', 'n_tokens', 'n_terms'))
        writer.writeheader()
        writer.writerows(stats)

def get_model(sentences, min_count, window_size, vector_size, max_vocab_size):
    ''' prepare a Word2Vec model and build its vocabulary '''
    model = Word2Vec(
        min_count=min_count,
        window=window_size,
        vector_size=vector_size,
        max_vocab_size=max_vocab_size
    )
    model.build_vocab(sentences)
    return model

def get_vectors_and_stats(model, sentences, n_tokens, independent):
    ''' return the word vectors of a model, and statistics on the number of terms and tokens '''
    if independent:
        vocab = model.wv.index_to_key
        n_terms = len(vocab)
        output_vectors = model.wv
    else:
        cv = CountVectorizer(analyzer=lambda x: x)
        doc_term = cv.fit_transform(sentences)
        cv_vocab = cv.get_feature_names_out()
        n_terms = len(cv_vocab)
        n_tokens = doc_term.sum()
        vectors = model.wv
        vocab = list(set(vectors.index_to_key).intersection(set(cv_vocab)))
        # restrict the KeyedVectors to only those in the vocab of this time slice
        output_vectors = KeyedVectors(model.vector_size)
        output_vectors.add_vectors(
            vocab, [vectors[word] for word in vocab])
    return output_vectors, n_terms, n_tokens
    

if __name__ == '__main__':
    generate_models()
