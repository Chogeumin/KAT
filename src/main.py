from Lecture2Vec import Lecture2Vec
from Predictor import Predictor
from konlpy.tag import Twitter
import sys, glob

if __name__ == "__main__":
    if (sys.argv[1] == 'T'):
        model = Lecture2Vec()
        model.build(vocab=sys.argv[2], corpus=sys.argv[3], distinct=1)
    pred = Predictor(distinct=1)
    t = Twitter()
    positive = t.nouns(sys.argv[4])
    negative = []
    if (len(sys.argv) == 6):
        negative = t.nouns(sys.argv[5])
    print(pred.top_rank(positive=positive, negative=negative, threshold=0))
