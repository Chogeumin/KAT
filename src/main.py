import Lecture2Vec as l2v
import Predictor as pd

if __name__ == "__main__":
    model = l2v.Lecture2Vec(university="hyu", campus="erica")
    model.build("corpus/train", 100)
    pred = pd.Predictor("wv/100vectors.bin")
    lectures = pred.top_rank("정보검색론", 0)
    print(lectures)