# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os

class MySentences(object):
    """Read text files by lines from directory and return string list. string list isn't pos tagging just split sentence.

    Parameters
    ----------
    dirname : str
        folder path which contains text files. Default is "./"

    """
    def __init__(self, dirname="./"):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

class Lecture2Vec(object):
    """Class for build lecture vector.
        In this version, you can only build lecture vectors at Hanyang University ERICA campus(hyu, erica).

        Parameters
        ----------
        university : str
            University Abbreviation. Default is "hyu"

        campus : str
            Campus name. Default is "erica"


    """

    def build(self, vocab, corpus, distinct, name):
        """Build lecture vector from corpus using word2vec in gensim library. corpus to make vocab and to train are different.
        
        Parameters
        ----------
        corpus : str
            Corpus list used to train word2vec model

        distinct : int
            if distinct is 1, train corpus and vocabulary corpus is different.
            if distinct is 0, train corpus and vocabulary corpus is same.

        """
        if (distinct == True):
            vocab = MySentences(vocab)
            train = MySentences(corpus)

            model = Word2Vec(min_count=1)
            model.build_vocab(sentences=vocab)
            model.train(sentences=train,
                        total_examples=model.corpus_count,
                        epochs=model.iter)

            word_vectors = model.wv
        elif (distinct == False):
            vocab = []
            train = MySentences(corpus)

            model = Word2Vec(sentences=train)

            word_vectors = model.wv

        del vocab, train, model
        word_vectors.save_word2vec_format(name, binary=True)
