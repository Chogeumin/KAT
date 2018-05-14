# -*- coding: utf-8 -*-

from konlpy.tag import Twitter
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import glob, sys, os


# 구현에 필요한 함수
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


def print_initial_message(width, message):
    horizontal_line = "-" * int(width)
    print(horizontal_line)
    print(message)
    print(horizontal_line)


def progress(cnt, length):
    now = int(cnt / length * 100)
    bar = '■' * now + '-' * (100 - now)
    sys.stdout.write('\r%s: |%s| %d%s' % ("Progress", bar, now, '%'))


# Console 첫 화면
width = os.popen('stty size', 'r').read().split()[1]
message = "\nKorean lecture recommendation Artificial intelligence Technology\nver 1.012\n"
print_initial_message(width, message)


# Corpus 제작.
print("Loading Corpus")
sentences = MySentences('data/')

# CBOW 모델 제작 및 저장.
print("Training model")
size = int(sys.argv[1])
model = Word2Vec(sentences, size=size)
word_vectors = model.wv
del sentences

print("Save model")
file_name = "wv/" + str(size) + "vectors.bin"
word_vectors.save_word2vec_format(file_name, binary=True)
