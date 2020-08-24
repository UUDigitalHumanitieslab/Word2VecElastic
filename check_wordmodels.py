from gensim.models import KeyedVectors

def test_model(modelfile):
    """ Given a .w2v modelfile, run tests to check if the word models make sense """
    model = KeyedVectors.load_word2vec_format(modelfile, binary=True)
    print(model.most_similar('water'))
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))