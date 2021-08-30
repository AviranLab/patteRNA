import numpy as np
from sklearn.linear_model import LogisticRegression


class LBC:
    def __init__(self):
        self.features = ['c-score', 'BCE', 'MEL']
        self.coefs = np.array([[1.28279679, -0.03173247, 0.13668635]])
        self.intercept = np.array([0.68955016])
        self.classes = np.array([0, 1])

        self.classifier = LogisticRegression()
        self.classifier.coef_ = self.coefs
        self.classifier.intercept_ = self.intercept
        self.classifier.classes_ = self.classes

    def apply_classifier(self, scores):
        if scores:
            xf = np.array([[score[feature] for feature in self.features] for score in scores])
            list(map(lambda score, lr_prob: write_prob(score, lr_prob[1]), scores, self.classifier.predict_proba(xf)))
        else:
            pass

    def get_positive_prob(self, features):
        lr_prob = self.classifier.predict_proba(features)
        return lr_prob[0][1]


def write_prob(score, prob):
    score['Prob(motif)'] = prob
