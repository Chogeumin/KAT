from gensim.models import KeyedVectors

class Predictor:
    def __init__(self, model):
        """
        Init Model directory
        """
        self.model = KeyedVectors.load_word2vec_format(model, binary=True)
    
    def top_rank(self, lecture, threshold):
        lecture_list = []
        for lecture in self.model.most_similar(lecture):
            if (lecture[1] >= threshold):
                lecture_list.append(lecture)
        return lecture_list

    def predict(self, neighbors, threshold):
        score = 0

        return score
