import os
from os.path import basename, join, splitext
import pickle
from glob import glob

from gensim.models import KeyedVectors


def check_path(path):
    """Check if path exists and if it does not, create it."""
    if not os.path.isdir(path):
        os.mkdir(path)

def correct_vocab(model_folder):
    """Utility function to trim down the vocabulary of the KeyedVectors
    to that from the CountVectorizer """
    os.chdir(model_folder)
    kvs = glob('*.w2v')
    for k in kvs:
        model = KeyedVectors.load_word2vec_format(k, binary=True)
        vocab = model.key_to_index.keys()
        name_scheme = splitext(k)[0]
        cv_vocab_file = '{}_vocab.pkl'.format(name_scheme)
        with open(cv_vocab_file, 'rb') as f:
            cv_vocab = pickle.load(f)
        out_vocab = list(set(vocab).intersection(set(cv_vocab)))
        os.remove(cv_vocab_file)
        with open(cv_vocab_file, 'wb') as f:
            pickle.dump(out_vocab, f)
        out_vocab_text = '{}_vocab.txt'.format(name_scheme)
        with open(out_vocab_text, 'w+') as f:
            f.writelines(out_vocab)

def sentences_to_lowercase(input_folder, output_folder):
    old_files = glob('{}/*.pkl'.format(input_folder))
    check_path(output_folder)
    for sen_file in old_files:
        output_file = join(output_folder, basename(sen_file))
        with open(output_file, 'wb') as f_out:
            with open(sen_file, 'rb') as f_in:
                sentences = pickle.load(f_in)
                for sen in sentences:
                    new_sen = [s.lower() for s in sen]
                    pickle.dump(new_sen, f_out)

            