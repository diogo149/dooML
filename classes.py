"""
TODO
    -

Table of Contents
    -GenericObject
    -MachineWrapper
    -PredictReshaper
    -PrefitMachine
    -PickleMachine
    -SparseFiltering
    -NumericalToCategorical
    -CategoricalToNumerical
    -FittedClustering
    -FittedMiniBatchKMeans
    -FittedKMeans
"""

import numpy as np
import re

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans, KMeans


from utils import quick_save, is_categorical
from helper import sparse_filtering
from helper.gap_statistic import gap_statistic


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

    def _pre_init(self, kwargs):
        pass  # override me

    def _post_init(self, kwargs):
        pass  # override me

    def __validate_args(self):
        for arg in self.__get_class_arg("_required_args", ()):
            assert arg in self.__dict__, arg

    def __get_class_arg(self, name, default=None):
        return self.__class__.__dict__.get(name, default)


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
            self.clustering = FittedMiniBatchKMeans(self.min_clusters)

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

    _default_args = dict(min_clusters=1)

    def fit(self, X, y=None):
        num_clusters = max(gap_statistic(X), self.min_clusters)
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


class FittedMiniBatchKMeans(FittedClustering):
    _clustering = MiniBatchKMeans


class FittedKMeans(FittedClustering):
    _clustering = KMeans
