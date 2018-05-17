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
    
    def __init__(self, university="hyu", campus="erica"):
        self.university = university
        self.campus = campus

    def build(self, corpus, size):
        """Build lecture vector from corpus using word2vec in gensim library.
        
        Parameters
        ----------
        corpus : str
            Corpus list used to train word2vec model

        szie : int
            Vector dimension. Please care to use. This parameter yields memory lack.

        """
        vocab = MySentences(corpus + "/" + self.university + "/" + self.campus)
        train = MySentences(corpus + "/train")

        model = Word2Vec(size=size)
        model.build_vocab(sentences=vocab)
        model.train(sentences=train)

        word_vectors = model.wv
        del vocab, train, model

        word_vectors.save_word2vec_format("wv/" + str(size) + "vectors.bin", binary=True)
