from gensim.models import KeyedVectors
from sklearn import neighbors
import numpy as np

class Predictor:
    def __init__(self, model):
        self.model = KeyedVectors.load_word2vec_format(model, binary=True)
        self.label = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    def top_rank(self, lecture, threshold): 
        """
        @lecture: source lecture name.
        @threshold: if cosin similarity is under than threshold, then it is regarded sparse similarity with source lecture.

        top_rank function takes lecture name then find top 10 most similar using gensim library. If there exists lecture whichs similarity is under threshhold, then that lecture is thrown.
        """

        lecture_list = []
        for lecture in self.model.most_similar(lecture):
            if (lecture[1] >= threshold):
                lecture_list.append(lecture)
        
        return lecture_list

    def predict(self, target, lectures, scores):
        """
        @target: target lecture vector
        @sroces: lecture scroe list
        @lectures: lecture vector list

        predict function takes lecture vector list and lecture score list. These lists are associated idx. Using 2 lists, first calculate distance between target and neighbors. Then using scikit-learn library, find score.
        """
        # TODO: 코드 구현
        clf = neighbors.KNeighborsClassifier(n_neighbors=len(lectures), weights='distance')
        clf.fit(lectures, scores)
        score = clf.predict(target)[0]

        return score
        