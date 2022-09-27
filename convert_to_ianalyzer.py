# convert output from training to current i-analyzer format

from gensim.models import KeyedVectors
import numpy as np
import os
import re
import pickle
from collect_sentences import DataCollector
from generate_models import MIN_COUNT

MODELS_DIR = '../models'
OUTPUT_DIR = '../pickled_models'
LANGUAGE = 'english'


def import_keyed_vectors(modelfile):
    if modelfile.endswith('.model'):
        word2vec = KeyedVectors.load(modelfile)
        keyedvectors = word2vec.wv
        del word2vec
    else:
        keyedvectors = KeyedVectors.load_word2vec_format(modelfile, binary=True)

    return keyedvectors

def get_years(filename):
    pattern = r'(\d{4})-(\d{4})'
    match = re.search(pattern, filename)
    start_year = int(match.group(1))
    end_year = int(match.group(2))

    return start_year, end_year

def generate_docs(start_year, end_year):
    for filename in os.listdir(os.path.join(MODELS_DIR, 'source_data')):
        if not 'vectorizer' in filename:
            year = int(re.search(r'\d{4}', filename).group(0))
            if year >= start_year and year < end_year:
                path = os.path.join(MODELS_DIR, 'source_data', filename)
                with open(path, 'rb') as datafile:
                    data = pickle.load(datafile)
                    for doc in data:
                        for word in doc:
                            # yield 'documents' of one word each
                            # so we can use min_df to enforce minimum term frequency
                            yield word

def build_transformer(start_year, end_year):
    cv = DataCollector.build_vectorizer(LANGUAGE, min_df=MIN_COUNT)
    docs = generate_docs(start_year, end_year)
    cv.fit(docs)
    vocab_size = len(cv.vocabulary_)
    print('{}-{}: {} words'.format(start_year, end_year, vocab_size))

    return cv

def build_matrix(keyedvectors, transformer):
    vectors = [keyedvectors.get_vector(key) for key in transformer.get_feature_names()]
    matrix = np.transpose(np.array(vectors))
    return matrix

def test_index(keyedvectors, transformer, matrix):
    term = 'people'

    transformer_index = transformer.vocabulary_[term]
    saved_vector = matrix[:, transformer_index]

    originaL_vector = keyedvectors.get_vector(term)
    assert np.array_equal(originaL_vector, saved_vector)


def import_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    start_year, end_year = get_years(filename)
    keyedvectors = import_keyed_vectors(path)
    transformer = build_transformer(start_year, end_year)
    matrix = build_matrix(keyedvectors, transformer)

    test_index(keyedvectors, transformer, matrix)

    return {
        'transformer': transformer,
        'svd_ppmi': matrix,
        'start_year': start_year,
        'end_year': end_year
    }

# convert complete model

def import_complete_model():
    filename = next(f for f in os.listdir(MODELS_DIR) if f.endswith('.model'))
    model = import_model(filename)
    return model

def save_complete_model():
    model = import_complete_model()
    output_path = os.path.join(OUTPUT_DIR, 'complete.pkl')
    with open(output_path, 'wb') as outfile:
        pickle.dump(model, outfile)


# convert binned models


def import_all_binned_models():
    models = [import_model(filename) for filename in os.listdir(MODELS_DIR) if filename.endswith('.w2v')]
    sorted_models = list(sorted(models, key= lambda model: model['start_year']))
    return sorted_models

def save_binned_models():
    models = import_all_binned_models()
    output_path = os.path.join(OUTPUT_DIR, 'binned.pkl')
    with open(output_path, 'wb') as outfile:
        pickle.dump(models, outfile)


if __name__ == '__main__':
    save_complete_model()
    save_binned_models()