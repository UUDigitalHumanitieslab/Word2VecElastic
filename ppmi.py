from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import svds
import numpy as np


def count_words(corpus, stop_words=None, min_df=2, max_features=30000):
    transformer = CountVectorizer(
        min_df=min_df, 
        stop_words=stop_words, 
        max_features=max_features,
        strip_accents='ascii')
    counts = transformer.fit_transform(corpus)
    return transformer, counts


def ppmi(counts):
    """ given a sparse matrix of where each word occurs in each document,
    return the positive pointwise mutual information
    Code partially based on hyperwords by Omar Levy
    """
    row_coefs = counts.sum(axis=0) # summing up all occurrences of each word
    col_coefs = counts.sum(axis=1) # summing up all occurrences of each document
    sum_total = row_coefs.sum() # all words occurrences summed
    norm_row = dok_matrix((len(row_coefs.T), len(row_coefs.T)))
    norm_col = dok_matrix((len(col_coefs), len(col_coefs)))
    norm_row.setdiag((1/row_coefs).T)
    norm_col.setdiag(1/col_coefs) # may give runtime warning due to divides by zero
    ppmi = counts
    ppmi = ppmi.dot(norm_row.tocsr())
    ppmi = norm_col.tocsr().dot(ppmi)
    ppmi = sum_total * ppmi
    ppmi.data = np.log(ppmi.data)
    ppmi.data = np.clip(ppmi.data, 0, None)
    return ppmi


def svd_ppmi(ppmi, n_features=200):
    """ given a sparse matrix of ppmi values, perform singular value decomposition
    n_features determines how many singular values to use
    """
    number_of_features = min(ppmi.shape[0]-1, n_features)
    left, singular, right = svds(ppmi, number_of_features)
    return right

