from Lecture2Vec import Lecture2Vec
from Predictor import Predictor
from konlpy.tag import Twitter
import sys

if __name__ == "__main__":
    # model = Lecture2Vec(university=sys.argv[1], campus=sys.argv[2])
    # model.build(corpus=sys.argv[3], distinct=int(sys.argv[4]))
    pred = Predictor(distinct=2)
    t = Twitter()
    positive = t.nouns(sys.argv[1])
    print(pred.top_rank(lecture=positive, threshold=0))
