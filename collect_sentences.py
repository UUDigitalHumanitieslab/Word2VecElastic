import time
import pickle
import os

from elasticsearch import Elasticsearch
from nltk.tokenize import PunktSentenceTokenizer


from util import check_path

import logging
logger = logging.getLogger(__name__)

try:
    from config import ES_HOST, ES_PORT
except:
    ES_HOST = 'localhost'
    ES_PORT = 9200

node = {'host': ES_HOST,
        'port': ES_PORT,
    }
kwargs = {
    'max_retries': 15,
    'retry_on_timeout': True,
    'timeout': 180
}

try:
    from config import API_ID, API_KEY, CERTS_LOCATION
    node['scheme'] = 'https'
    kwargs['ca_certs'] = CERTS_LOCATION
    kwargs['api_key'] = (API_ID, API_KEY)
except:
    node['scheme'] = 'http'
      
es = Elasticsearch([node], **kwargs)


class DataCollector():
    '''
    class which can be used as iterator over data stored in Elasticsearch index
    in the first pass, will collect data from Elasticsearch and save it as pickle files
    in subsequent passes, will return data from those pickle files
    also pickles a CountVectorizer, which can be later used to reproduce the analyzer
    - index: the name of the alias or index from which to collect data
    - start_year: first year from which to collect data
    - end_year: last year from which to collect data
    - analyzer: analyzer for the corpus, which does the following:
        -remove stop words
        -remove numbers
        -lowercase
    - field: field from which to collect data
    - source_directory: directory in which source data is saved
    '''
    def __init__(self, index, start_year, end_year, analyzer, field, date_field, source_directory):
        self.index = index
        self.start_year = start_year
        self.end_year = end_year
        self.field = field
        self.date_field = date_field
        self.extra_filter = None # None for now, could be used for e.g. removing newspaper adverts
        self.generator = self.set_generator_function()
        self.analyzer = analyzer
        self.source_directory = source_directory

    def __iter__(self):
        self.set_generator_function()
        return self
        
    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result
    
    def set_generator_function(self):
        self.generator = self.get_sentences()
    
    def get_sentences(self):
        for year in range(self.start_year, self.end_year):
            filename = self.get_pickle_filename(year)
            if os.path.exists(filename):
                with open(filename, 'rb') as source_file:
                    eof = False
                    while not eof:
                        try:
                            yield pickle.load(source_file)
                        except:
                            eof = True
            else:
                sentences = self.get_sentences_for_year(year)
                if not sentences:
                    continue
                with open(filename, 'wb') as text_file:
                    for sentence in sentences:
                        analyzed = self.analyzer(sentence)
                        pickle.dump(analyzed, text_file)
                        yield analyzed
    
    def get_pickle_filename(self, year):
        check_path(self.source_directory)
        return os.path.join(self.source_directory, '{}-{}.pkl'.format(self.index, year))

    def get_sentences_for_year(self, year):
        '''Return list of lists of strings.
        Return list of sentences in given year.
        Each sentence is a string.'''
        docs = self.get_documents_for_year(year)
        if not docs:
            return None
        sentences = []
        for doc in docs:
            doc_tok = self.tokenize_sentences(doc)
            if doc_tok:
                sentences.extend(doc_tok)
        return sentences
    
    def get_documents_for_year(self, year):
        '''Retrieves a list of documents for a year specified.'''
        min_date = str(year)+"-01-01"
        max_date = str(year)+"-12-31"
        search_body = self.get_es_body(min_date, max_date)
        docs = None
        for retry in range(10):
            try:
               docs = es.search(index=self.index, body=search_body, size=1000, scroll="60m", track_total_hits=True)
            except Exception as e:
                logger.warning(e)
                time.sleep(10)
        if not docs:
            es.clear_scroll(scroll_id='_all')
            return None
        content = [result['_source'][self.field] for result in docs['hits']['hits']]
        total_hits = docs['hits']['total']['value']
        if total_hits == 0:
            es.clear_scroll(scroll_id='_all')
            return None
        scroll_id = docs['_scroll_id']
        while len(content)<total_hits:
            scroll_id = docs['_scroll_id']
            try:
                docs = es.scroll(scroll_id=scroll_id, scroll="60m")
            except Exception as e:
                logger.warning(e)
                time.sleep(10)
                docs = es.search(index=self.index, body=search_body, size=1000, scroll="60m")
                content = [result['_source'][self.field] for result in docs['hits']['hits']]
                continue
            content.extend([result['_source'][self.field] for result in docs['hits']['hits']])
        es.clear_scroll(scroll_id=scroll_id)
        return content

    def tokenize_sentences(self, body):
        """Transform a single news paper article into a list of sentences (each
        sentence represented by a string)."""
        sent_tokenizer = PunktSentenceTokenizer()
        if isinstance(body, str):
            return sent_tokenizer.tokenize(body)
        else:
            logger.warning('{} is not a string'.format(body))

    def get_es_body(self, min_date, max_date):
        body = { 
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range" : {
                                self.date_field : {
                                    "gte" : min_date,
                                    "lte" : max_date
                                }
                            }
                        }
                    ]
                }
        }}
        if self.extra_filter:
            body['query']['bool']['filter'].append(
                {
                    "terms": {
                        self.extra_filter
                    }
                }
            )
        return body
