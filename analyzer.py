from nltk.corpus import stopwords
import spacy
from spacy.tokens import Doc
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex, compile_suffix_regex

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
        # modify tokenizer so that it doesn't split on hyphens,
        # but does treat a hyphen as a suffix
        infix_re = compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_re.finditer
        suffixes = self.nlp.Defaults.suffixes + [r'''-+$''',]
        suffix_regex = compile_suffix_regex(suffixes)
        self.nlp.tokenizer.suffix_search = suffix_regex.search
    
    def preprocess(self, input_string):
        # apply analysis pipeline
        doc = self.nlp(input_string)
        
        def select_token(token):
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
        
        output = [select_token(token) for token in doc]
        return output
    

