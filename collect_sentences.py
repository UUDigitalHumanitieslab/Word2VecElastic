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

class SentencesFromElasticsearch(object):
    def __init__(self, minYear, maxYear, index):
        self.minYear = minYear
        self.maxYear = maxYear
        self.index = index
    
    def __iter__(self):
        return self

    def __next__(self):
        for year in range(self.minYear, self.maxYear):
            tic = time.perf_counter()
            documents = getDocumentsForYear(year, self.index)
            toc = time.perf_counter()
            print('Fetching documents for year {} took {} seconds'.format(
                year, toc - tic))
            for doc in documents:
                sentences = _getSentencesInArticle(doc)
                if not sentences:
                    continue
                with open('sentences.pkl', 'ab') as f:
                    for sentence in sentences:
                        output = _prepareSentence(sentence)
                        if output:
                            pickle.dump(output, f)
                            return output


class SentencesFromPickle(object):
    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        with open('sentences.pkl', 'rb') as f:
            while True:
                self.counter += 1
                try:
                    sentence = pickle.load(f)
                    return sentence
                except EOFError:
                    os.remove('sentences.pkl')
                    print(self.counter)
                    raise StopIteration


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
    return { "query": {
        "bool": {
            "filter": [
                {
                    "range" : {
                        "date" : {
                            "gte" : min_date,
                            "lte" : max_date
                        }
                    }
                },
                {
                    "terms": {
                        "circulation": ["Landelijk", "Regionaal/lokaal"]
                    }
                } 
            ]
        }
    }}

def getDocumentsForYear(year, index):
    '''Retrieves a list of documents for a year specified.'''
    min_date = str(year)+"-01-01"
    max_date = str(year)+"-12-31"
    search_body = getSearchBody(min_date, max_date)
    for retry in range(10):
        try:
           docs = es.search(index=index, body=search_body, size=1000, scroll="60m")
        except Exception as e:
            logger.warning(e)
            time.sleep(10)
    if not docs:
        return None
    content = [result['_source']['content'] for result in docs['hits']['hits']]
    total_hits = docs['hits']['total']['value']
    scroll_id = docs['_scroll_id']
    while len(content)<total_hits:
        scroll_id = docs['_scroll_id']
        try:
            docs = es.scroll(scroll_id=scroll_id, scroll="60m")
        except Exception as e:
            es.clear_scroll(scroll_id=scroll_id)
            logger.warning(e)
            time.sleep(10)
            docs = es.search(index=index, body=search_body, size=1000, scroll="60m")
            content = [result['_source']['content'] for result in docs['hits']['hits']]
            continue
        content.extend([result['_source']['content'] for result in docs['hits']['hits']])
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
        doc_tok = _getSentencesInArticle(doc.decode('utf-8'))
        if doc_tok:
            sentences.extend(doc_tok)
    final_sentences = [_prepareSentence(sentence) for sentence in sentences]
    print(year, len(final_sentences))
    return final_sentences


def _getSentencesInArticle(body):
    """Transform a single news paper article into a list of sentences (each
    sentence represented by a string)."""
    sent_tokenizer = nltk.punkt.PunktSentenceTokenizer()
    if isinstance(body, str):
        sentences = sent_tokenizer.tokenize(body)
        return sentences
    else:
        print(body)


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
    if word in _dutchStopWords:
        return False
    elif len(word)<3:
        return False
    elif _removePunctuation.search(word):
        return False
    else:
        return True