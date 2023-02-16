import re
from nltk.corpus import stopwords
import spacy
from spacy.tokens import Doc
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex, compile_suffix_regex

spacy_models = {
    'english': "en_core_web_sm",
    'german': "de_core_news_sm",
    'french': "fr_core_news_sm"
}

class Analyzer(object):
    def __init__(self, language, lemmatize):
        self.lemmatize = lemmatize
        self.language = language
        self.stopword_list = stopwords.words(language)
        model = spacy_models.get(self.language)
        self.nlp = spacy.load(model)
    
    def preprocess(self, input_string):
        # apply analysis pipeline
        doc = self.nlp(input_string)
        # there are some suffixes that indicate we don't want hyphen splitting
        exceptions = ['anti', 'e', 'extra', 'inter', 'neo', 'non', 'post', 'pre', 'pro']
        word_indices = [
            token.i for token in doc if token.text in exceptions
        ]
        with doc.retokenize() as retokenizer:
            for index in word_indices:
                retokenizer.merge(doc[index:index+3])
        output = [self.select_token(token).lower() for token in doc if self.select_token(token)]
        return output

    def select_token(self, token):
        exclude_conditions = [
            token.is_punct,
            token.is_currency,
            token.is_stop,
            token.is_digit
        ]
        if any(exclude_conditions):
            pass
        elif self.lemmatize:
            return token.lemma_
        else:
            return token.text
 