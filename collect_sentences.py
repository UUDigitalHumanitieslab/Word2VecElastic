from elasticsearch import Elasticsearch

import string
import re
import nltk
from nltk.corpus import words, wordnet, stopwords

node = {'host': 'elastic.dhlab.hum.uu.nl',
        'port': 9200}
es = Elasticsearch([node], timeout=60, max_retries=10, retry_on_timeout=True)
_englishWords = set(w.lower() for w in words.words())
_englishStopWords = set(stopwords.words('english'))
_removePunctuation = re.compile('[%s]' % re.escape(string.punctuation))

class SentencesFromElasticsearch(object):
    def __init__(self, minYear, maxYear, index):
        self.minYear = minYear
        self.maxYear = maxYear
        self.index = index
    def __iter__(self):
        for year in range(self.minYear, self.maxYear):
            documents = getDocumentsForYear(year, self.index)
            for doc in documents:
                sentences = _getSentencesInArticle(doc)
                if not sentences:
                    continue
                for sentence in sentences:
                    output = _prepareSentence(sentence)
                    if output:
                        yield output

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
                }
            ]
        }
    }}

def getNumberArticlesForTimeInterval(startY, endY, index):
    min_date = str(startY)+"-01-01"
    max_date = str(endY)+"-12-31"
    search_body = getSearchBody(min_date, max_date)
    docs = es.search(index=index, body=search_body, size=0)
    total_hits = docs['hits']['total']
    return total_hits


def getDocumentsForYear(year, index):
    '''Retrieves a list of documents for a year specified.'''
    min_date = str(year)+"-01-01"
    max_date = str(year)+"-12-31"
    search_body = getSearchBody(min_date, max_date)
    docs = es.search(index=index, body=search_body, size=1000, scroll="1m")
    content = [result['_source']['content'] for result in docs['hits']['hits']]
    total_hits = docs['hits']['total']['value']
    scroll_id = docs['_scroll_id']
    while len(content)<total_hits:
        if 'scroll_id' in docs:
            scroll_id = docs['_scroll_id']
        docs = es.scroll(scroll_id=scroll_id, scroll="1m")
        content.extend([result['_source']['content'] for result in docs['hits']['hits']])
    es.clear_scroll(scroll_id=scroll_id)
    return content


def getSentencesForYear(year):
    '''Return list of lists of strings.
    Return list of sentences in given year.
    Each sentence is a list of words.
    Each word is a string.'''
    docs = getDocumentsForYear(year)
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