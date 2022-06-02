#!/usr/bin/env python
"""Generate a set of time shifting models from the given period of years. Each
one of the models spans a given number of years.

Usage:
  runGenerateModels.py --y0=<y0> --yN=<yN> --nYears=<years> --outDir=<dir> [--step=<years>] --index=<index>

Options:
  --y0 <y0>         First year in the generated models
  --yN <yN>         Last year in the generated models
  --nYears <years>  Number of years per model
  --outDir <dir>    Directory where models will be writen to
  --step <years>    Step between start year of generated models [default: 1]
  --index <index>   Which index to use for generating models
"""
from gensim.models import Word2Vec

from docopt import docopt
from collect_sentences import sentences_from_elasticsearch, \
    SentencesFromPickle, getNumberArticlesForTimeInterval
from util import checkPath
import csv
from os.path import isfile
import os

import logging
logging.basicConfig(filename='models.log', level=logging.WARNING, filemode='a', datefmt='%Y-%m-%d %H:%M:%S', 
    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

def generateModels(y0, yN, yearsInModel, stepYears, modelFolder, index):
    """Generate time shifting w2v models on the given time range (y0 - yN).
    Each model contains the specified number of years (yearsInModel). The start
    year of each new model is set to be stepYears after the previous model.
    Resulting models are saved on modelFolder.
    """
    checkPath(modelFolder)
    csv_filename = 'count-{}.csv'.format(index)
    fieldnames = ['year', 'articles', 'tokens', 'words', 'min_count']
    if not isfile(csv_filename):
        with open(csv_filename, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

    for year in range(y0, yN - yearsInModel + 1, stepYears):
        startY = year
        endY = year + yearsInModel
        logger.warning('Calculating years: {}-{}'.format(startY, endY))
        total_count = getNumberArticlesForTimeInterval(startY, endY, index)
        logger.warning('Total number of articles: '+str(total_count))
        if isfile('sentences{}-{}.pkl'.format(startY, endY)):
            sentences = SentencesFromPickle(startY, endY)
        else:
            sentences = sentences_from_elasticsearch(startY, endY, index)
        tokens, words = count_tokens_words(sentences)
        logger.warning('Tokens: {}, Words: {}'.format(tokens, words))
        min_count = int(words / 200000)
        with open(csv_filename, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writerow({'year': year, 
            'articles': total_count, 'tokens': tokens, 'words': words,
            'min_count': min_count})
        modelName = '{}/{}_{}.w2v'.format(modelFolder, startY, endY)
        sentences = SentencesFromPickle()
        logger.warning('Building model: '+modelName)
        if year == y0:
            # this is the initial model
            model = Word2Vec(sentences, min_count=100, workers=14, epochs=5)
            model.save(modelName)
        else:
            previousModel = '{}/{}_{}.w2v'.format(modelFolder, startY-step, endY-step)
            model = Word2Vec.load(previousModel)
            model.build_vocab(sentences, update=True)
            model.train(
                sentences,
                total_examples=model.corpus_count,
                start_alpha=model.alpha,
                end_alpha=model.min_alpha,
                epochs=model.iter
            )
            model.save(modelName)
        logger.warning('Saving to {}'.format(modelName))
    

def count_tokens_words(sentences):
    token_count = 0
    words = set()
    for sentence in sentences:
        token_count += len(sentence)
        words.update(sentence)
    return token_count, len(words)


if __name__ == '__main__':
    args = docopt(__doc__)
    yearsInModel = int(args['--nYears'])
    stepYears = int(args['--step'])
    outDir = args['--outDir']
    y0 = int(args['--y0'])
    yN = int(args['--yN'])
    index = args['--index']

    generateModels(y0, yN, yearsInModel, stepYears, outDir, index)
