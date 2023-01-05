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
from sklearn.feature_extraction.text import CountVectorizer

from collect_sentences import DataCollector
from analyzer import Analyzer
from util import check_path

import logging
logging.basicConfig(filename='models.log', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %H:%M:%S', 
    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

MIN_COUNT = 50
N_DIMS = 128

@click.command()
@click.option('-i', '--index', help="Elasticsearch index name from which to request the training data", required=True)
@click.option('-s', '--start_year', help="Year from which to start training", required=True)
@click.option('-e', '--end_year', help="Year until which to continue training", required=True)
@click.option('n', '--n_years', help="Number of years each model should span", default=10)
@click.option('-sh', '--shift_years', help="Shift between models", default=5)
@click.option('-d', '--directory', help="Directory in which the models should be saved", required=True)
@click.option('-f', '--field', help="Field from which to extract training data", default='content')
@click.optoin('-l', '--language', help="Language of the training data", default='english')
@click.option('-lem', '--lemmatize', help="Whether or not to perform lemmatization", default=False, is_flag=True)
@click.option('-mc', '--min_count', help="Minimum count of a given word to be included in a model", default=MIN_COUNT)
@click.option('-vs', '--vector_size', help="The size of the embedding vectors", default=N_DIMS)
@click.option('-mv', '--max_vocab_size', help="Limit the size of the vocab, i.e., prune")
def generate_models(
        index,
        start_year,
        end_year,
        n_years,
        shift_years,
        directory,
        field,
        language,
        lemmatize,
        min_count,
        vector_size,
        max_vocab_size):
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
    analyzer = Analyzer(language, lemmatize).preprocess
    cv = CountVectorizer(analyzer=analyzer)
    sentences = DataCollector(index, start_year, end_year, analyzer, field, model_folder)
    full_model_name = '{}_{}_{}_full'.format(index, start_year, end_year)
    full_model_file =  '{}.model'.format(full_model_name)
    if not os.path.exists(join(model_folder, full_model_file)):
        model = Word2Vec(min_count=min_count, vector_size=vector_size, max_vocab_size=max_vocab_size)
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
            'n_terms': len(cv_vocab)})
        
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
    generate_models()
