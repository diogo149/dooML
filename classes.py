"""Table of Contents
    -GenericObject
    -ContextManagerObject
    -INSTANCE_GETATTR
    -SingletonContextManagerObject
    -Trial
    -FeatureCache
    -MachineCache
    -PredictReshaper
    -NumericalToCategorical
    -CategoricalToNumerical
    -FittedClustering
    -fitted_minibatchkmeans
    -fitted_kmeans

    -BinningMachine
    -NumpyBinningMachine
    -classification_sampling_machine
    -regression_sampling_machine
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import re
import warnings

from time import time
from copy import deepcopy
from types import MethodType
from functools import partial
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans, KMeans

from storage import quick_write, quick_save, quick_load, quick_exists, machine_cache
from utils import is_categorical, smart_hash, random_seed
from helper import gap_statistic

# these are imported so that they can be imported from this file
from helper.binning_machine import BinningMachine, NumpyBinningMachine
from helper.sampling_machine import classification_sampling_machine, regression_sampling_machine


class GenericObject(object):

    """ generic object with magic functions implemented
    """

    def __init__(self, **kwargs):
        self.__dict__.update(self.__get_class_arg("_default_args", {}))
        self._pre_init(kwargs)
        self.__dict__.update(kwargs)
        self._post_init(kwargs)
        self.__validate_args()

    def __repr__(self):
        regex = r"<class '.*\.(\w+)'>"
        return "{}(**{})".format(re.search(regex, repr(self.__class__)).group(1), repr(self.__dict__))

    def __str__(self):
        return repr(self)

    def __call__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def _pre_init(self, kwargs):
        pass  # override me

    def _post_init(self, kwargs):
        pass  # override me

    def __validate_args(self):
        for arg in self.__get_class_arg("_required_args", ()):
            assert arg in self.__dict__, arg

    def __get_class_arg(self, name, default=None):
        return self.__class__.__dict__.get(name, default)


class ContextManagerObject(GenericObject):

    """ generic object for easy use with context managers
    """

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class INSTANCE_GETATTR(type):

    """ metaclass that redirects non-static calls to the child class' INSTANCE object
    """

    def __getattribute__(cls, name):
        func = type.__getattribute__(cls, name)
        if isinstance(func, MethodType):
            func = partial(func, cls.INSTANCE)
        return func


class SingletonContextManagerObject(ContextManagerObject):

    """ Made for inheritance for singletons that redirects non-static calls to the class' INSTANCE object. Also allows the class' INSTANCE object to be initialized with a context manager
    """

    __metaclass__ = INSTANCE_GETATTR

    def __enter__(self):
        assert self.__class__.__dict__.get('INSTANCE') is None, self.__class__.__dict__.get('INSTANCE')
        self.__class__.INSTANCE = self
        return super(SingletonContextManagerObject, self).__enter__()

    def __exit__(self, *args, **kwargs):
        self.__class__.INSTANCE = None
        return super(SingletonContextManagerObject, self).__exit__(*args, **kwargs)


class Trial(SingletonContextManagerObject):

    """ Class that allows one to only fit machines when training, and to use trained machines while testing by changing one function call. Use with a context manager:

        with Trial.train("description..."):
            # train classifiers, etc.
    """
    DIRECTORY = "trials"
    _required_args = ('description', 'train_mode')

    def _post_init(self, kwargs):
        assert isinstance(self.description, str)
        self.filename = np.abs(hash(self.description))
        if self.train_mode:
            assert not quick_exists(Trial.DIRECTORY, self.filename, "txt"), self.filename
            quick_write(Trial.DIRECTORY, self.filename, "")
            quick_save(Trial.DIRECTORY, self.filename, None)
            self.clf_times = []
            self.clfs = []
        else:
            self.__dict__ = quick_load(Trial.DIRECTORY, self.filename).__dict__
            self.train_mode = False
        random_seed(self.filename)

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
            trained_clf = self.clfs.pop(0)
            assert isinstance(trained_clf, clf.__class__)
            return trained_clf

    def log(self, **kwargs):
        self(**kwargs)

    def train_mode(self):
        return self.train_mode

    def filename(self):
        return self.filename


class FeatureCache(object):

    def __init__(self, df):
        self._rows = df.shape[0]
        self._directory = smart_hash(df)
        self.raw = df

    def validate(self, df, is_safe=True):
        assert isinstance(df, pd.DataFrame), df.__class__
        assert len(df.shape) == 2, df.shape
        assert df.shape[0] == self._rows, (df.shape, self._rows)
        if is_safe:
            df.astype(np.float)  # this should fail if dataframe isn't numeric
            assert not np.any(pd.isnull(df))
            assert not np.any(np.isinf(df))

    def _put(self, is_safe, name, func, *args, **kwargs):
        result = func(*args, **kwargs)
        self.validate(result, is_safe=is_safe)
        result = result.astype(np.float) if is_safe else result
        quick_save(self._directory, name, result)
        return result

    def put_unsafe(self, name, func, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._put(False, name, func, *args, **kwargs)

    def put(self, name, func, *args, **kwargs):
        return self._put(True, name, func, *args, **kwargs)

    def get(self, name):
        return quick_load(self._directory, name)

    def cache(self, name, func, *args, **kwargs):
        try:
            return self.get(name)
        except:
            print("Feature Cache Miss: {}".format(name))
            return self.put(name, func, *args, **kwargs)


class MachineCache(GenericObject):

    """ class for caching machines, based on their hash. the purpose of this class is to cache the hashes of the arrays.
    """

    def _post_init(self, kwargs):
        # need to use two lists instead of a dictionary, to avoid hashing every time
        self.input = []
        self.hashes = []

    def cache(self, clf, X, y=None):
        item = (X, y)
        try:
            idx = self.input.index(item)
        except ValueError:
            idx = len(self.input)
            self.input.append(item)
            self.hashes.append("{}_{}".format(smart_hash(X), smart_hash(y)))
        filename = self.hashes[idx]
        return machine_cache(filename, clf, X, y)


class NumericalToCategorical(GenericObject):

    """Takes in a clustering classifier in order to convert numerical features into categorical.
    """

    _default_args = dict(clustering=None, min_clusters=2, verify=True)

    def _post_init(self, kwargs):
        if kwargs['clustering'] is None:
            self.clustering = fitted_minibatchkmeans(self.min_clusters)

    def fit(self, X, y=None):
        self._verify(X, self.verify)
        reshaped = X.reshape(-1, 1)
        self.clustering.fit(reshaped)

    def transform(self, X):
        self._verify(X, False)
        reshaped = X.reshape(-1, 1)
        result = self.clustering.predict(reshaped)
        assert result.shape == X.shape
        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _verify(self, X, verify):
        if verify:
            assert not is_categorical(X)
        else:
            assert isinstance(X, np.ndarray)
            assert len(X.shape) == 1


class CategoricalToNumerical(GenericObject):

    _default_args = dict(dimensionality_reducer=None, verify=True)

    def _post_init(self, kwargs):
        if kwargs['dimensionality_reducer'] is None:
            self.dimensionality_reducer = RandomizedPCA(1)
        self.binarizer = LabelBinarizer()

    def fit(self, X, y=None):
        self._verify(X, self.verify)
        binarized = self.binarizer.fit_transform(X)
        self.dimensionality_reducer.fit(binarized)

    def transform(self, X):
        self._verify(X, False)
        binarized = self.binarizer.transform(X)
        result = self.dimensionality_reducer.transform(binarized).flatten()
        assert X.shape == result.shape
        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _verify(self, X, verify):
        if verify:
            assert is_categorical(X)
        else:
            assert isinstance(X, np.ndarray)
            assert len(X.shape) == 1


class FittedClustering(GenericObject):

    """ clustering algorithm that uses gap statistic to automatically determine how many clusters to use
    """

    _required_args = ('clustering',)
    _default_args = dict(min_clusters=1)

    def fit(self, X, y=None):
        num_clusters = max(gap_statistic.gap_statistic(X), self.min_clusters)
        clustering = self.__get_class_arg("_clustering")
        self.clf = clustering(num_clusters)
        self.clf.fit(X)

    def transform(self, X):
        return self.clf.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def predict(self, X):
        return self.clf.predict(X)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


def fitted_minibatchkmeans(min_clusters=1):
    return FittedClustering(clustering=MiniBatchKMeans, min_clusters=min_clusters)


def fitted_kmeans(min_clusters=1):
    return FittedClustering(clustering=KMeans, min_clusters=min_clusters)

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    with Trial.train("this is my description"):
        x = np.random.randn(10, 10)
        y = np.random.randn(10)
        Trial.fit(LinearRegression(), x, y)
        Trial.log(this="bananas")

    with Trial.test("this is my description"):
        clf = Trial.fit(LinearRegression())
        print(clf.predict(np.random.randn(10, 10)) - np.random.randn(10))
        Trial.log(hello="world")

    """ check the generated text file and the pickle
    """
