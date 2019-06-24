#!/usr/bin/env python
"""Generate a set of time shifting models from the given period of years. Each
one of the models spans a given number of years.

Usage:
  runGenerateModels.py --y0=<y0> --yN=<yN> --nYears=<years> --outDir=<dir> [--step=<years>]

Options:
  --y0 <y0>         First year in the generated models
  --yN <yN>         Last year in the generated models
  --nYears <years>  Number of years per model
  --outDir <dir>    Directory where models will be writen to
  --step <years>    Step between start year of generated models [default: 1]
"""
import gensim

import logging
logging.basicConfig(filename='models.log', level=logging.WARNING, filemode='w+')
logger = logging.getLogger(__name__)

from docopt import docopt
from collect_sentences import SentencesFromElasticsearch, getNumberArticlesForTimeInterval
from util import checkPath


def generateModels(y0, yN, yearsInModel, stepYears, modelFolder):
    """Generate time shifting w2v models on the given time range (y0 - yN).
    Each model contains the specified number of years (yearsInModel). The start
    year of each new model is set to be stepYears after the previous model.
    Resulting models are saved on modelFolder.
    """
    checkPath(modelFolder)

    for year in range(y0, yN - yearsInModel + 1, stepYears):
        startY = year
        endY = year + yearsInModel
        modelName = modelFolder + '/%d_%d.w2v' % (year, year + yearsInModel)
        vocabName = modelName.replace('.w2v', '.vocab.w2v')
        logger.warning('Building model: '+modelName)
        total_count = getNumberArticlesForTimeInterval(startY, endY)
        logger.warning('Total number of articles: '+str(total_count))
        sentences = SentencesFromElasticsearch(startY, endY)
        tokens, words = count_tokens_words(sentences)
        logger.warning('Tokens: {}, Words: {}'.format(tokens, words))
        sentences = SentencesFromElasticsearch(startY, endY)
        model = gensim.models.Word2Vec(min_count=100)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        print('...saving')
        model.init_sims(replace=True)
        model.wv.save_word2vec_format(modelName, fvocab=vocabName, binary=True)
    

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

    generateModels(y0, yN, yearsInModel, stepYears, outDir)
