#!/usr/bin/env python
"""Generate a set of time shifting models from the given period of years. Each
one of the models spans a given number of years.

Usage:
  runGenerateModels.py --start=<startYear> --end=<endYear> --n=<years> --out=<dir> [--step=<years>] --index=<index> [--lang=<language>] --field=<field> [--shift=<years>]

Options:
  --start <startYear>   First year in the generated models
  --end <endYear>   Last year in the generated models
  --n <years>   Number of years per model
  --out <dir>   Directory where models will be writen to
  --shift <years>   Step between start year of generated models [default: 1]
  --index <index>   Which index to use for generating models
  --lang <language> Which language stopword list to use [default: english]
  --field <field>   Which field in the Elasticsearch index is used for extracting sentences
"""
import csv
from os.path import join
import os

from docopt import docopt
from gensim.models.word2vec import Word2Vec

from collect_sentences import DataCollector
from util import check_path

import logging
logging.basicConfig(filename='models.log', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %H:%M:%S', 
    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

MIN_COUNT = 10

def generate_models(start_year, end_year, years_in_model, model_folder, index, field, shift_years=1, language='english'):
    """Generate time shifting w2v models on the given time range (start_year - end_year).
    Each model contains the specified number of years (years_in_model). The start
    year of each new model is set to be shift_years after the previous model.
    Resulting models are saved on model_folder.
    """
    check_path(model_folder)
    sentences = DataCollector(index, start_year, end_year, language, field, model_folder)
    full_model_name = '{}-{}-{}-full.model'.format(index, start_year, end_year)
    if not os.path.exists(join(model_folder, full_model_name)):
        model = Word2Vec(min_count=MIN_COUNT, alpha=0.1)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(os.path.join(model_folder, full_model_name))

    for year in range(start_year, end_year - years_in_model + 1, shift_years):
        start = year
        end = year + years_in_model
        model = Word2Vec.load(join(model_folder, full_model_name))
        model_name = '{}-{}-{}.w2v'.format(index, start, end)
        logger.info('Building model: '+ model_name)
        sentences = DataCollector(index, start, end, language, field, model_folder)
        model.train(sentences, total_examples=len(list(sentences)), epochs=model.epochs)
        logger.info('Saving to {}'.format(model_name))
        # init_sims precomputes the L2 norm, model cannot be trained further after this step
        model.init_sims(replace=True)
        model.wv.save_word2vec_format(join(model_folder, model_name), binary=True)
    

def count_tokens_words(sentences):
    token_count = 0
    words = set()
    for sentence in sentences:
        token_count += len(sentence)
        words.update(sentence)
    return token_count, len(words)


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

    generate_models(
        start_year,
        end_year,
        years_in_model,
        model_folder,
        index,
        field,
        shift_years,
        language
    )
