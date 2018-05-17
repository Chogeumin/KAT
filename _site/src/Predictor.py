from gensim.models import KeyedVectors
from sklearn import neighbors
import numpy as np

class Predictor:
    def __init__(self, distinct):
        """ Using Predictor class, you can get at most 10 top rank which is consine similiarty over threshold. Futhermore, can predict similar lecture using kNN

        Parameter
        ---------
        distinct : int
            if you build vector to use different model, then distinct is 1.
            else, distinct is 0.
            
        """

        self.model = KeyedVectors.load_word2vec_format("data/vectors" + str(distinct) + ".bin", binary=True)
        self.label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def top_rank(self, lecture, threshold): 
        """ top_rank function takes lecture name then find top 10 most similar using gensim library. If there exists lecture whichs similarity is under threshhold, then that lecture is thrown.

        Parameter
        ---------
        lecture : list
            source lecture name tokenized by Twitter.

        threshold : float
            if cosin similarity is under than threshold, then it is regarded sparse similarity with source lecture.

        
        """

        lecture_list = []
        for lecture in self.model.most_similar(positive=lecture):
            if (lecture[1] >= threshold):
                lecture_list.append(lecture)
        
        return lecture_list

    def predict(self, target, lectures, scores):
        """predict function takes lecture vector list and lecture score list. These lists are associated idx. Using 2 lists, first calculate distance between target and neighbors. Then using scikit-learn library, find score.

        Parameter
        ---------
        target : list
            target lecture vector

        sroces : list
            lecture scroe list

        lectures : list
            lecture vector list
        
        """
        clf = neighbors.KNeighborsClassifier(n_neighbors=len(lectures), weights='distance')
        clf.fit(lectures, scores)
        
        score = clf.predict(target)[0]

        return score
