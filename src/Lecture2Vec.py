# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os

class MySentences(object):
    """
    Read text files by lines from directory and return string list. string list isn't pos tagging just split sentence.
    """
    
    def __init__(self, dirname="./"):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

class Lecture2Vec(object):
    """
    Class for build lecture vector.
    In this version, you can only build lecture vectors at Hanyang University ERICA campus(hyu, erica).
    """

    def build(self, vocab, corpus, name):
        """
        Build lecture vector from corpus using word2vec in gensim library. corpus to make vocab and to train are different.
        """

        vocab = MySentences(vocab)
        train = MySentences(corpus)
        model = Word2Vec(size=100,
                        min_count=1)
        
        model.build_vocab(sentences=vocab)
        model.train(sentences=train,
                    total_examples=model.corpus_count,
                    epochs=model.iter)

        word_vectors = model.wv

        word_vectors.save_word2vec_format(name, binary=True)
