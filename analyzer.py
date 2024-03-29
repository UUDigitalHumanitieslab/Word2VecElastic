import spacy

import logging
logging.basicConfig(filename='analysis.log', level=logging.WARNING, filemode='a', datefmt='%Y-%m-%d %H:%M:%S', 
    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

spacy_models = {
    'english': "en_core_web_sm",
    'german': "de_core_news_sm",
    'french': "fr_core_news_sm",
    'swedish': "sv_core_news_sm",
    'dutch': "nl_core_news_sm",
    'finnish': "fi_core_news_lg"
}

class Analyzer(object):
    def __init__(self, language, lemmatize):
        self.lemmatize = lemmatize
        self.language = language
        model = spacy_models.get(self.language)
        self.nlp = spacy.load(model)
    
    def preprocess(self, input_string):
        # apply analysis pipeline
        doc = self.nlp(input_string)
        # there are some prefixes that indicate we don't want hyphen splitting
        exceptions = ['anti', 'e', 'extra', 'inter', 'neo', 'non', 'post', 'pre', 'pro', 'ultra']
        # first, check that the prefixes are indeed connected with a hyphen
        prefixes = [
            pref for pref in exceptions if '{}-'.format(pref) in input_string.lower()
        ]
        if len(prefixes):
            # get all indices where a potential hyphenated prefix could be located
            # e.g., 'We do not support neo-liberalism' -> index would be -3
            # as we merge at least three tokens, drop indices after -3
            word_indices = [
                token.i for token in doc[:-3] if token.text.lower() in prefixes
            ]
            with doc.retokenize() as retokenizer:
                for index in word_indices:
                    if doc[index+1].text == '-':
                        try:
                            retokenizer.merge(doc[index:index+3])
                        except Exception as e:
                            logger.error(input_string, doc[index:index+3])
                            continue
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
