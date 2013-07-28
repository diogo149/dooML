"""Table of Contents
    -GenericObject
    -ContextManagerObject
    -INSTANCE_GETATTR
    -SingletonContextManagerObject
    -MachineWrapper
    -PredictReshaper
    -PrefitMachine
    -PickleMachine
    -SparseFiltering
    -NumericalToCategorical
    -CategoricalToNumerical
    -FittedClustering
    -fitted_minibatchkmeans
    -fitted_kmeans
    -TransformPredictMachine
    -kmeans_linear_model
"""

import numpy as np
import re

from types import MethodType
from functools import partial
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.linear_model import ElasticNet


from storage import quick_save
from utils import is_categorical
from helper import sparse_filtering, gap_statistic


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
        self._trial.update(**kwargs)

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

    __metaclass__ = INSTANCE_GETATTR

    def __enter__(self):
        assert self.__class__.__dict__.get('INSTANCE') is None, self.__class__.__dict__.get('INSTANCE')
        self.__class__.INSTANCE = self
        return super(SingletonContextManagerObject, self).__enter__()

    def __exit__(self, *args, **kwargs):
        self.__class__.INSTANCE = None
        return super(SingletonContextManagerObject, self).__exit__(*args, **kwargs)


class MachineWrapper(GenericObject):

    """ parent class of a series of classes meant to wrap machine learning algorithms
    """

    def __getattr__(self, name):
        return getattr(self.clf, name)


class PredictReshaper(MachineWrapper):

    """ Wraps a machine so that it's predict method returns a 2D array with a specified number of columns (defualt=1).
    """
    _default_args = {'predict_cols': 1, }

    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs).reshape(-1, self.predict_cols)


class PrefitMachine(MachineWrapper):

    """ Wraps a machine so that it's fit method doesn't change object state.
    """

    def fit(self, *args, **kwargs):
        pass


class PickleMachine(MachineWrapper):

    """ Wraps a machine so that it saves a pickle of the classifier after training.
    """

    def fit(self, *args, **kwargs):
        result = self.clf.fit(*args, **kwargs)
        quick_save("pickle_machine", self.name, self.clf)
        return result


class SparseFiltering(GenericObject):

    """ Machine for calling sparse filtering algorithm.
    """

    _default_args = dict(N=1000)

    def fit(self, X, y=None):
        self.W = sparse_filtering.sparseFiltering(self.N, X.T)

    def transform(self, X):
        return sparse_filtering.feedForwardSF(self.W, X.T)


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


class TransformPredictMachine(GenericObject):

    _required_args = ('transformer', 'predicter')

    def fit(self, X, y):
        tmp = self.transformer.fit_transform(X, y)
        self.predicter.fit(tmp, y)

    def predict(self, X):
        tmp = self.transformer.transform(X)
        return self.predicter.predict(tmp)


def kmeans_linear_model(n_clusters=2, alpha=1.0, l1_ratio=0.5, fit_intercept=True, positive=False):
    transformer = MiniBatchKMeans(n_clusters=n_clusters, compute_labels=False, random_state=None)
    predicter = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, positive=positive)
    return TransformPredictMachine(transformer, predicter)
