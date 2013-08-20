import itertools
import time
import logging
import random
import numpy as np

from sklearn.linear_model import SGDClassifier


class SGDRank(object):

    """ Ranking predictor using stochastic gradient descent
    """

    def __init__(self, seconds=10):
        # self.clf = SGDRegressor()
        self.clf = SGDClassifier(loss='hinge')
        self.clf.fit_intercept = False
        self.clf.classes_ = np.array([-1, 1])
        self.seconds = seconds

    def fit(self, X, y):
        rows = X.shape[0]
        start_time = time.time()
        for i in itertools.count():
            idx1 = random.randint(0, rows - 1)
            idx2 = random.randint(0, rows - 1)
            y1, y2 = y[idx1], y[idx2]
            if y1 == y2:
                continue
            self.clf.partial_fit(X[idx1] - X[idx2], np.sign(y1 - y2))
            if time.time() - start_time > self.seconds:
                logging.debug("SGDRank completed {} iterations".format(i))
                return self

    def predict(self, X):
        return np.dot(X, self.clf.coef_.T)
