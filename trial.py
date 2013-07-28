import random
from classes import SingletonContextManagerObject
import numpy as np
from time import time

from copy import deepcopy
from storage import quick_write, quick_save, quick_load, quick_exists


class Trial(SingletonContextManagerObject):

    """ Class that allows one to only fit machines when training, and to use trained machines while testing by changing one function call. Use with a context manager:

        with Trial.train("description..."):
            # train classifiers, etc.
    """
    DIRECTORY = "trials"
    _default_args = ('description', 'train_mode')

    def _post_init(self, kwargs):
        assert isinstance(self.description, str)
        self.filename = hash(self.description)
        self.clfs = []
        if self.train_mode:
            assert not quick_exists(Trial.DIRECTORY, self.filename, "txt")
            quick_write(Trial.DIRECTORY, self.filename, "")
            quick_save(Trial.DIRECTORY, self.filename, None)
        else:
            self.__dict__ = quick_load(Trial.DIRECTORY, self.filename).__dict__
        np.random.seed(self.filename)
        random.seed(self.filename)

    @staticmethod
    def train(description):
        return Trial(description=description, train_mode=True)

    @staticmethod
    def test(description):
        return Trial(description=description, train_mode=False)

    def close(self):
        if self.train_mode:
            quick_write(Trial.DIRECTORY, self.filename, self)
            quick_save(Trial.DIRECTORY, self.filename, self)

    def fit(self, clf, *args, **kwargs):
        """ returns a fitted classifier
        """
        if self.train_mode:
            start_time = time()
            clf.fit(*args, **kwargs)
            self.clf_times.append(time() - start_time)
            self.clfs.append(deepcopy(clf))
            return clf
        else:
            return self.clfs.pop(0)

    def log(self, **kwargs):
        self(**kwargs)
