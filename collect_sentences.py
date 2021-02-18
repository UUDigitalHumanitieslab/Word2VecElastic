import logging
logger = logging.getLogger(__name__)
import time
import pickle
from os.path import isfile
import string
import re

from elasticsearch import Elasticsearch
import pandas as pd
import spacy
import nltk
from tqdm import tqdm

import config

node = {'host': config.ES_HOST,
        'port': config.ES_PORT}
es = Elasticsearch(
    [node], 
    http_auth=(config.ES_USER, config.ES_PASSWORD),
    timeout=180
)
nlp = spacy.load("en_core_web_trf")
sent_tokenizer = nltk.punkt.PunktSentenceTokenizer()

def sentences_from_elasticsearch(minYear, maxYear, index):
    for year in range(minYear, maxYear):
        tic = time.perf_counter()
        documents = getDocumentsForYear(year, index)
        toc = time.perf_counter()
        print('Fetching documents for year {} took {} seconds.'.format(
            year, toc - tic))
        if not documents:
            continue
        first_doc = True
        for doc in tqdm(documents):
            analyzed_document = _analyzeDocument(doc)
            places = analyzed_document.pop('places')
            df = pd.DataFrame()
            df = df.append(analyzed_document, ignore_index=True)
            df.to_csv('documents.csv', mode='a', header=first_doc, index=False)
            if places:
                place_output = [{'year': year, 'place': p['text']} for p in places]
                df = pd.DataFrame(place_output)
                df.to_csv('places.csv', mode='a', header=first_doc, index=False)
            first_doc = False
        
            


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
    content = [result for result in docs['hits']['hits']]
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
            content = [result for result in docs['hits']['hits']]
            continue
        content.extend([result for result in docs['hits']['hits']])
    es.clear_scroll(scroll_id=scroll_id)
    return content


def _analyzeDocument(doc):
    """ Analyze the document using spaCy. """
    sentences = sent_tokenizer.tokenize(doc['_source']['content'])
    entities = []
    lemmas = []
    places = []
    for sentence in sentences:
        sent_analyzed = nlp(sentence)
        entities.extend(
            [{'text': ent.text, 'label': ent.label_} for ent in sent_analyzed.ents]
        )
        # get all non-stopword and non-punctuation lemmas
        lemmas.extend(
            [token.lemma_ for token in sent_analyzed if not token.is_stop and token.is_alpha]
        )
    places = [ent for ent in entities if ent['label']=='GPE']
    return {'id': doc['_id'], 'date': doc['_source']['date'], 'entities': entities, 'lemmas': lemmas, 'places': places}