from gensim.models import KeyedVectors

class Predictor:
    def __init__(self, std_id, lecture):
        self.std_id = std_id
        self.lecture = lecture
