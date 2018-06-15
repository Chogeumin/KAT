from Lecture2Vec import Lecture2Vec
from Predictor import Predictor

from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import re
import pandas as pd
import matplotlib.pyplot as plt

import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--build", type=bool, default=False, help="모델 제작 True/False")
    parser.add_argument('-v', "--vocab", type=str, default="data/vocab/", help="디렉토리 위치 corpus/<folder_name>")
    parser.add_argument('-c', "--corpus", type=str, default="corpus/", help="디렉토리 위치 corpus/<folder_name >")
    parser.add_argument('-p', "--pred", type=bool, default=False, help="예측기 사용 여부 True/False")
    parser.add_argument('-n', '--name', type=str, default='data/auto.bin', help="워드벡터 저장 위치 data/<binary file name>.bin")
    parser.add_argument('-l', '--lecture', type=str, default="data/lectures.txt")
    parser.add_argument('-t', '--threshold', type=float, default=0.0)
    parser.add_argument('-r', '--rectangular', type=bool, default=False)
    args = parser.parse_args()

    for key in vars(args).keys():
        print(key, ":", vars(args)[key])

    if (args.build == True):
        model = Lecture2Vec()
        model.build(vocab=args.vocab, corpus=args.corpus, name=args.name)
        
    if (args.rectangular == True):
        model = KeyedVectors.load_word2vec_format(args.name, binary=True)
        
        pred = Predictor(name=args.name)
        vocab_file = open('data/vocab/en_department.txt')
        vocab = vocab_file.read().splitlines()

        X = []
        for v in vocab:
            tokens = v.split()
            vector = pred.get_vector(words=tokens)
            X.append(vector)

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

        print("x")
        print(df['x'])

        print("y")
        print(df['y'])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['x'], df['y'])

        for word, pos in df.iterrows():
            ax.annotate(word, pos)
        plt.show()

    if (args.pred == True):
        pred = Predictor(name=args.name)

        lecture_file = open(args.lecture, 'r')
        lecture_list = lecture_file.read().splitlines()
        lectures = []
        for l in lecture_list:
            tokens = l.split()
            vector = pred.get_vector(words=tokens)
            lectures.append((l, vector))
    
        most_similars = {}
        for lecture in lectures:
            most_similars[lecture[0]] = pred.most_similar(target=lecture, lectures=lectures, threshold=args.threshold)

        for k in most_similars.keys():
            print(k)
            print(most_similars[k])
            print("-" * 10)
