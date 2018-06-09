from gensim.models import KeyedVectors
from sklearn import neighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Predictor:
    def __init__(self, name):
        """
        Using Predictor class, you can get at most 10 top rank which is consine similiarty over threshold. Futhermore, can predict similar lecture using kNN
        """

        self.model = KeyedVectors.load_word2vec_format(name, binary=True)
        self.label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    

    def top_rank(self, positive, negative = "", threshold = 0): 
        """
        top_rank function takes lecture name then find top 10 most similar using gensim library. If there exists lecture whichs similarity is under threshhold, then that lecture is thrown.
        """

        lecture_list = []
        for lecture in self.model.most_similar(positive=positive, negative=negative):
            if (lecture[1] >= threshold):
                lecture_list.append(lecture)
        
        return lecture_list


    def predict(self, target, lectures, scores):
        """
        predict function takes lecture vector list and lecture score list. These lists are associated idx. Using 2 lists, first calculate distance between target and neighbors. Then using scikit-learn library, find score.
        """

        clf = neighbors.KNeighborsClassifier(n_neighbors=len(lectures), weights='distance')
        clf.fit(lectures, scores)
        
        score = clf.predict(target)[0]

        return score


    def get_vector(self, words):
        vector_length = len(self.model.wv[words[0]])
        sum_vector = np.zeros(vector_length, dtype=float)
        for word in words:
            sum_vector = np.add(sum_vector, self.model.wv[word])

        return sum_vector
    

    def most_similar(self, target, lectures, threshold):
        result = []
        
        for lecture in lectures:
            if (target[0] == lecture[0]):
                continue
            
            numerator = np.sum(np.multiply(target[1], lecture[1]))
            denominator = np.sqrt(np.sum(np.multiply(target[1], target[1])) * np.sum(np.multiply(lecture[1], lecture[1])))
            similarity = numerator / denominator
            
            if similarity > threshold:
                # result.append([lecture[0], similarity, lecture[1]])
                result.append([lecture[0], similarity])
        
        result.sort(key=lambda x: x[1], reverse=True)
        
        if (len(result) > 5):
            result = result[0:5]

        return result
