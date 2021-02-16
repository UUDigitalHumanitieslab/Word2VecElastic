import logging
logger = logging.getLogger(__name__)
import time
import pickle
from os.path import isfile
import csv

from elasticsearch import Elasticsearch

import string
import re
import nltk
from nltk.corpus import words, wordnet, stopwords
import spacy

import config

node = {'host': config.ES_HOST,
        'port': config.ES_PORT}
es = Elasticsearch(
    [node], 
    http_auth=(config.ES_USER, config.ES_PASSWORD),
    timeout=180
)
nlp = spacy.load("en_core_web_trf")

_englishWords = set(w.lower() for w in words.words())
_englishStopWords = set(stopwords.words('english'))
_dutchStopWords = set(stopwords.words('dutch'))
_removePunctuation = re.compile('[%s]' % re.escape(string.punctuation))

def sentences_from_elasticsearch(minYear, maxYear, index):
    if not isfile(config.CSV_FILE):
        with open(config.CSV_FILE, "w+") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=('text', 'label'))
            csv_writer.writeheader()
    for year in range(minYear, maxYear):
        tic = time.perf_counter()
        documents = getDocumentsForYear(year, index)
        toc = time.perf_counter()
        print('Fetching documents for year {} took {} seconds.'.format(
            year, toc - tic))
        if not documents:
            continue
        
        for doc in documents:
            analyzed_doc = _analyzeDocument(doc)
            if not analyzed_doc:
                continue
            with open('documents.pkl', 'ab') as f:
                pickle.dump(analyzed_doc, f)
            places = [ent.update({'year': year}) for ent in analyzed_doc['entities'] if ent['label']=='GPE']
            if places:
                with open(config.CSV_FILE, "a") as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=('text', 'label', 'year'))
                    csv_writer.writerows(places)
            


class SentencesFromPickle():
    def __init__(self):
        self.generator = self.generator_function()

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
        with open('sentences.pkl', 'rb') as file:
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


def getSearchBody(min_date, max_date):
    return {
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
            logger.warning(e)
            time.sleep(10)
            docs = es.search(index=index, body=search_body, size=1000, scroll="60m")
            content = [(result['_source']['id'], result['_source']['content']) for result in docs['hits']['hits']]
            continue
        content.extend([(result['_source']['id'], result['_source']['content']) for result in docs['hits']['hits']])
    es.clear_scroll(scroll_id=scroll_id)
    return content


def getAnalyzedDocuments(docs):
    ''' Return list of documents for each year.
    Each document is returned with its id, and its analyzed content.
    '''
    analyzed_docs = []
    for doc in docs:
        doc_tok = _analyzeDocument(doc[1].decode('utf-8'))
        if doc_tok:
            entities = [{'text': ent.text, 'label': ent.label_} for ent in doc_tok.ents]
            # get all non-stopword and non-punctuation lemmas
            lemmas = [token.lemma_ for token in doc_tok if not token.is_stop and token.is_alpha]
            analyzed_docs.append({'id': doc[0], 'entities': entities, 'lemmas': lemmas})
    return analyzed_docs


def _analyzeDocument(doc):
    """ Analyze the document using spaCy. """
    doc_analyzed = nlp(doc)
    return doc_analyzed


def _isValidWord(word):
    """Determine whether a word is valid. A valid word is a valid english
    non-stop word."""
    if word in _englishStopWords:
        return False
    elif word in _englishWords:
        return True
    elif wordnet.synsets(word):
        return True
    else:
        return False