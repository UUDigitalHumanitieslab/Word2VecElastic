from nltk.corpus import stopwords
import spacy
from spacy.tokens import Doc
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

spacy_models = {
    'English': "en_core_web_sm"
}

class Analyzer(object):
    def __init__(self, language, lemmatize):
        self.lemmatize = lemmatize
        self.language = language
        self.stopword_list = stopwords.words(language)
        model = spacy_models.get(self.language)
        self.nlp = spacy.load(model)
        # modify tokenizer so that it doesn't split on hyphens
        infix_re = compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_re.finditer
    
    def preprocess(self, input_string):
        # apply analysis pipeline
        doc = self.nlp(input_string)
        return [select_token(token) for token in doc]
        
    def select_token(self, token):
        if token.is_punct or token.is_currency or token.is_stop or token.is_digit:
            pass
        if self.lemmatize:
            return token.lemma_
        return token.text
    

