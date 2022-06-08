import logging
logger = logging.getLogger(__name__)
import time
import pickle
import os

from elasticsearch import Elasticsearch

import string
import re
import nltk
from nltk.corpus import words, wordnet, stopwords

node = {'host': 'elastic.dhlab.hum.uu.nl',
        'port': 9200}
es = Elasticsearch([node], timeout=180)
_englishWords = set(w.lower() for w in words.words())
_englishStopWords = set(stopwords.words('english'))
_dutchStopWords = set(stopwords.words('dutch'))
_removePunctuation = re.compile('[%s]' % re.escape(string.punctuation))

def sentences_from_elasticsearch(minYear, maxYear, index):
    for year in range(minYear, maxYear):
        tic = time.perf_counter()
        documents = getDocumentsForYear(year, index)
        toc = time.perf_counter()
        print('Fetching documents for year {} took {} seconds.'.format(
            year, toc - tic))
        if not documents:
            continue
        
        for doc in documents:
            sentences = _getSentencesInDocument(doc)
            if not sentences:
                continue
            with open('sentences{}-{}.pkl'.format(minYear, maxYear), 'ab') as f:
                for sentence in sentences:
                    output = _prepareSentence(sentence)
                    if output:
                        pickle.dump(output, f)
                        yield output


class SentencesFromPickle(object):
    def __init__(self, minYear, maxYear):
        self.generator = self.generator_function()
        self.min_year = minYear
        self.max_year = maxYear

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result
    
    def generator_function(self):
        with open('sentences{}-{}.pkl'.format(self.min_year, self.max_year), 'rb') as file:
            while True:
                try:
                    yield pickle.load(file)
                except EOFError:
                    break


def getNumberArticlesForTimeInterval(startY, endY, index):
    min_date = str(startY)+"-01-01"
    max_date = str(endY)+"-12-31"
    search_body = getSearchBody(min_date, max_date)
    docs = es.search(index=index, body=search_body, track_total_hits=True, size=0)
    total_hits = docs['hits']['total']['value']
    return total_hits


def getMinYear(index):
    '''Retrieves the first year on the data set.'''
    body = {
        "aggs" : {
            "min_date" : { "min" : { "field" : "date" } }
        }
    }
    min_date = es.search(index=index, body=body, size=0)
    #return int(min_date['aggregations']['min_date']['value_as_string'][:4])
    return 1840 # returning fixed date for now

def getMaxYear(index):
    '''Retrieves the last year on the data set.'''
    body = {
        "aggs" : {
            "max_date" : { "max" : { "field" : "date" } }
        }
    }
    max_date = es.search(index=index, body=body, size=0)
    #return int(max_date['aggregations']['max_date']['value_as_string'][:4])
    return 1920 # returning fixed date for now


def getSearchBody(min_date, max_date):
    return {
        "_source": ["speech"],
        "query": {
            "range" : {
                "date" : {
                    "gte" : min_date,
                    "lte" : max_date
                }
            }
        }
    }

def getDocumentsForYear(year, index):
    '''Retrieves a list of documents for a year specified,
    scrolling in batches of 1000 documents '''
    min_date = str(year)+"-01-01"
    max_date = str(year)+"-12-31"
    search_body = getSearchBody(min_date, max_date)
    for retry in range(10):
        try:
           docs = es.search(
               index=index,
               body=search_body,
               size=1000,
               track_total_hits=True,
                scroll="60m"
            )
        except Exception as e:
            logger.warning(e)
            time.sleep(10)
    if not docs:
        return None
    content = [result['_source']['speech'] for result in docs['hits']['hits']]
    total_hits = docs['hits']['total']['value']
    scroll_id = docs['_scroll_id']
    while len(content)<total_hits:
        scroll_id = docs['_scroll_id']
        try:
            docs = es.scroll(scroll_id=scroll_id, scroll="60m")
        except Exception as e:
            # restart search in case of server timeout
            logger.warning(e)
            time.sleep(10)
            docs = es.search(index=index, body=search_body, size=1000, scroll="60m")
            content = [result['_source']['speech'] for result in docs['hits']['hits']]
            continue
        content.extend([result['_source']['speech'] for result in docs['hits']['hits']])
    es.clear_scroll(scroll_id=scroll_id)
    return content


def getSentencesForYear(year, index):
    '''Return list of lists of strings.
    Return list of sentences in given year.
    Each sentence is a list of words.
    Each word is a string.'''
    docs = getDocumentsForYear(year, index)
    sentences = []
    for doc in docs:
        doc_tok = _getSentencesInDocument(doc.decode('utf-8'))
        if doc_tok:
            sentences.extend(doc_tok)
    final_sentences = [_prepareSentence(sentence) for sentence in sentences]
    logger.info(year, len(final_sentences))
    return final_sentences


def _getSentencesInDocument(document):
    """Transform a single document into a list of sentences (each
    sentence represented by a string)."""
    sent_tokenizer = nltk.punkt.PunktSentenceTokenizer()
    if isinstance(document, str):
        sentences = sent_tokenizer.tokenize(document)
        return sentences
    else:
        logger.error(body)


def _prepareSentence(sentence):
    """Document preparation a sentence. Document preparation
    consists of: removing punctuation, removing invalid words, lowe casing and
    splitting into individual words."""
    sentence = _removePunctuation.sub('', sentence)
    sentence = sentence.lower()
    sentence = sentence.split(' ')
    sentence = [w for w in sentence if _isValidWord(w)]
    if len(sentence) > 0:
        return sentence


def _isValidWord(word):
    """Determine whether a word is valid. A valid word is a dutch
    non-stop word."""
    if word in _englishStopWords:
        return False
    elif len(word)<3:
        return False
    elif _removePunctuation.search(word):
        return False
    else:
        return True