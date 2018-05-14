# -*- coding: utf-8 -*-
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import sys, os


# 함수
def print_initial_message(width, message, ver):
    horizontal_line = "-" * int(width)
    print(horizontal_line)
    print(message)
    print("ver %s\n" % ver)
    print(horizontal_line)


width = os.popen('stty size', 'r').read().split()[1]
message = "\nKAT"
ver = "1.001"
print_initial_message(width, message, ver)


# 벡터 읽어오기
size = int(sys.argv[1])
file_name = "wv/" + str(size) + "vectors.bin"
word_vectors = KeyedVectors.load_word2vec_format('wv/500vectors.bin', binary=True)

while (True):
    # 인자 받는 부분
    p = []
    n = []
    p_input = ""
    n_input = ""
    while (True):
        p_input = input("+ : ")
        if (p_input == ""):
            break
        p.append(p_input)
    while (True):
        n_input = input("- : ")
        if (n_input == ""):
            break
        n.append(n_input)

    if(len(p) == 0 & len(n) == 0):
        break

    for i in range(0, len(p)):
        if (i == 0):
            e = p[i]
        else:
            e += " + "
            e += p[i]
    if (len(n) > 0):
        for i in range(0, len(n)):
            e += " - "
            e += n[i]



    
    output = "\n입력된 수식 = " + e + "\n"
    print(output)

    for result in word_vectors.most_similar(positive=p, negative=n):
        print(result)

    print("-" * int(width))
