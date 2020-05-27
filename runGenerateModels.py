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
import gensim

from docopt import docopt
from collect_sentences import SentencesFromElasticsearch, getNumberArticlesForTimeInterval
from util import checkPath
import csv
from os.path import isfile

import logging
logging.basicConfig(filename='models.log', level=logging.WARNING, filemode='a')
logger = logging.getLogger(__name__)

def generateModels(y0, yN, yearsInModel, stepYears, modelFolder, index):
    """Generate time shifting w2v models on the given time range (y0 - yN).
    Each model contains the specified number of years (yearsInModel). The start
    year of each new model is set to be stepYears after the previous model.
    Resulting models are saved on modelFolder.
    """
    checkPath(modelFolder)
    csv_filename = 'count-{}.csv'.format(index)
    fieldnames = ['year', 'articles', 'tokens', 'words']
    if not isfile(csv_filename):
        with open(csv_filename, "a+") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

    for year in range(y0, yN - yearsInModel + 1, stepYears):
        startY = year
        endY = year + yearsInModel
        logger.warning('Calculating years: {}-{}'.format(startY, endY))
        modelName = modelFolder + '/%d_%d.w2v' % (year, year + yearsInModel)
        vocabName = modelName.replace('.w2v', '.vocab.w2v')
        print('Building model: ', modelName)
        total_count = getNumberArticlesForTimeInterval(startY, endY, index)
        sentences = SentencesFromElasticsearch(startY, endY, index)
        # count how many articles, tokens and words are in the model
        tokens, words = count_tokens_words(sentences)
        logger.warning('Tokens: {}, Words: {}'.format(tokens, words))
        info_dict = {'year': year, 'articles': total_count, 'tokens': tokens, 'words': words}
        with open(csv_filename, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writerow(info_dict)
        # build the actual model
        sentences = SentencesFromElasticsearch(startY, endY, index)
        model = gensim.models.Word2Vec(min_count=1)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        print('...saving')
        model.init_sims(replace=True)
        model.wv.save_word2vec_format(modelName, fvocab=vocabName, binary=True)


def count_tokens_words(sentences):
    """ count the number of tokens and words in the given time interval """
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
