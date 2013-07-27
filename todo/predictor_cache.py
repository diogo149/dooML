""" todo: look into using shelve as a virtual directory instead of an actual directory in the filesystem (to prevent the error from creating too many files)

    note: shelve doesn't play nice with multiprocessing
"""
import numpy as np

from copy import deepcopy

from utils import quick_save, quick_load, hash_numpy


class PredictorCache(object):

    """
    Object that stores the cache of the most recently used predictor in memory, as well as handles switching that cache with another on the filesystem when a different classifier is used. It is most efficient when using the same classifier. None of it's methods are meant to be used, and no instances of the class are meant to be instantiated. As a warning, a possibility of a hash collision may lead to withdrawing/using/overwriting the wrong classifier.
    """

    CACHE_DIRECTORY = "cache"
    clf_str = None
    cache_dict = dict()

    @staticmethod
    def cache(clf, x, y):
        x, y = PredictorCache._validate(x, y)
        PredictorCache._preprocess(clf)
        return PredictorCache._get(clf, x, y)

    @staticmethod
    def _preprocess(clf):
        clf_str = "".join([c for c in str(clf) if c.isalnum() or c in "=,()_"])
        if PredictorCache.clf_str != clf_str:
            PredictorCache._save()
            PredictorCache.clf_str = clf_str
            PredictorCache._load()

    @staticmethod
    def _get(clf, x, y):
        key = PredictorCache._key(x, y)
        try:
            return PredictorCache.cache_dict[key]
        except KeyError:
            new_clf = deepcopy(clf)
            new_clf.fit(x, y)
            PredictorCache.cache_dict[key] = new_clf
            return deepcopy(new_clf)

    @staticmethod
    def _save():
        if PredictorCache.clf_str is not None:
            quick_save(PredictorCache.CACHE_DIRECTORY, PredictorCache.clf_str, PredictorCache.cache_dict)

    @staticmethod
    def _load():
        try:
            PredictorCache.cache_dict = quick_load(PredictorCache.CACHE_DIRECTORY, PredictorCache.clf_str)
        except IOError:
            PredictorCache.cache_dict = {}

    @staticmethod
    def _key(x, y):
        return "_".join((str(x.shape), hash_numpy(x), hash_numpy(y)))

    @staticmethod
    def _validate(x, y):
        x = np.array(x)
        y = np.array(y)
        assert len(x.shape) == 2
        assert len(y.shape) == 1
        assert x.shape[0] == y.shape[0]
        return x, y

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        PredictorCache._save()


def pcache(clf, x, y):
    """ either withdraws the classifier trained with x and y from the cache, or trains a classifier and adds it to the cache
    """
    return PredictorCache.cache(clf, x, y)

""" an instance of the class, meant to be used for context managers:

    >>> with cache_cm:
    ...    # do stuff with the cache

    the benefit of using a context manager is that the cache gets saved on exit as well
"""
cache_cm = PredictorCache()

if __name__ == "__main__":
    from sklearn.linear_model import Ridge
    x = np.random.randn(500, 500)
    y = np.random.randn(500)
    with PredictorCache():
        clf = pcache(Ridge(), x, y)
    print ((clf.predict(x) - y) ** 2).mean()
