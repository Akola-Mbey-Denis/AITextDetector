from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
import gensim
import numpy as np
from gensim.models.doc2vec import TaggedDocument
import re
'''
This code was adapted from:
    https://www.kdnuggets.com/2018/11/multi-class-text-classification-model-comparison-selection.html/2

'''
def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled

def train_dbow(model,data, epochs = 30):
    for epoch in range(epochs):
        model.train(utils.shuffle([x for x in tqdm(data)]), total_examples=len(data), epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    return model

def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors
    


