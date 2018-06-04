from Lecture2Vec import Lecture2Vec
from Predictor import Predictor
import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--build", type=bool, default=False, help="모델 제작 True/False")
    parser.add_argument('-v', "--vocab", type=str, default="", help="디렉토리 위치 corpus/<folder_name>")
    parser.add_argument('-c', "--corpus", type=str, default="", help="디렉토리 위치 corpus/<folder_name >")
    parser.add_argument('-d', "--dist", type=bool, default=False, help="vocab, corpus 구분 True/False")
    parser.add_argument('-p', "--pred", type=bool, default=False, help="예측기 사용 여부 True/False")
    parser.add_argument('-n', '--name', type=str, default='data/auto.bin', help="워드벡터 저장 위치 data/<binary file name>.bin")
    parser.add_argument('-l', '--lecture', type=str, default="data/lectures.txt")
    parser.add_argument('-t', '--threshold', type=float, default=0.0)
    args = parser.parse_args()

    print("\nparser statement")
    print("build:\t", args.build)
    print("vocab:\t", args.vocab)
    print("corpus:\t", args.corpus)
    print("dist:\t", args.dist)
    print("pred:\t", args.pred)
    print("name:\t", args.name)
    print("t.hold:\t", args.threshold)
    print("\n")

    print("--- Result ---")
    if (args.build == True):
        model = Lecture2Vec()
        model.build(vocab=args.vocab, corpus=args.corpus, distinct=args.dist, name=args.name)

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
            print(most_similars[k][0])
            print("-" * 10)