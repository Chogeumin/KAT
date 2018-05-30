from Lecture2Vec import Lecture2Vec
from Predictor import Predictor
# from konlpy.tag import Twitter
# from konlpy.tag import Mecab
import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--build", type=bool, default=True, help="모델 제작 True/False")
    parser.add_argument('-v', "--vocab", type=str, default="corpus", help="디렉토리 위치 corpus/<folder_name>")
    parser.add_argument('-c', "--corpus", type=str, default="corpus", help="디렉토리 위치 corpus/<folder_name >")
    # parser.add_argument('-t', "--token", type=str, default="X", help="토크나이저 사용 여부 True/False")
    parser.add_argument('-d', "--dist", type=bool, default=False, help="vocab, corpus 구분 True/False")
    parser.add_argument('-p', "--pred", type=bool, default=False, help="예측기 사용 여부 True/False")
    parser.add_argument('-n', '--name', type=str, default='data/auto.bin', help="워드벡터 저장 위치 data/<binary file name>.bin")
    args = parser.parse_args()

    print("\nparser statement")
    print("build:\t", args.build)
    print("vocab:\t", args.vocab)
    print("corpus:\t", args.corpus)
    # print("token:\t", args.token)
    print("dist:\t", args.dist)
    print("pred:\t", args.pred)
    print("name:\t", args.name)
    print("\n")

    if (args.build == True):
        print("Build Model")
        model = Lecture2Vec()
        model.build(vocab=args.vocab, corpus=args.corpus, distinct=args.dist, name=args.name)

    print("\n\n")

    if (args.pred == True):
        print("Predict lecture")
        pred = Predictor(name=args.name)

        lecture_file = open('data/lecture.txt', 'r')
        lectures = lecture_file.readlines()
        for l in lectures:
            token = l.split()
            top_10 = pred.top_rank(positive=token)
            print(l, ":", top_10)