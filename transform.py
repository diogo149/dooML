"""Table of Contents:
    -GenericTransform
    -TransformFactory
    -SklearnBridge
    -InfinityRemover
    -NearZeroVarianceFilter
    -SparseFiltering
    -RowApply
    -ColApply
    -row_normalizer

"""
import numpy as np

from functools import partial

from utils import to2d
from classes import GenericObject
from helper import sparse_filtering


class GenericTransform(GenericObject):

    """ abstract class to keep the transform assumptions
    """

    @staticmethod
    def _verify1(X):
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2

    @staticmethod
    def _verify2(X, y, result=False):
        if result or y is not None:
            assert isinstance(y, np.ndarray)
            assert len(y.shape) == 2
            assert y.shape[0] == X.shape[0]

    def fit(self, X, y=None):
        GenericTransform._verify1(X)
        self._fit(X, y)
        GenericTransform._verify2(X, y)
        return self

    def _fit(self, X, y=None):
        pass

    def transform(self, X):
        GenericTransform._verify1(X)
        transformed = self._transform(X)
        GenericTransform._verify2(X, transformed, result=True)
        return transformed

    def _transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class TransformFactory(GenericTransform):

    """ Class that allows wrapping of functions into a transform. Assumes that the first input to the transform function is the output of the fit function
    """
    _required_args = ('fit_func', 'transform_func')

    def _fit(self, X, y=None):
        self.fit_params = self.fit_func(X, y)
        return self

    def _transform(self, X):
        return self.transform_func(self.fit_params, X)


class SklearnBridge(GenericTransform):

    """ class to wrap sklearn classifiers. tries predict_proba before predict.
    """

    _required_args = ('clf',)

    def _fit(self, X, y=None):
        if y is not None and y.shape[1] == 1:
            y = y.flatten()
        self.clf.fit(X, y)

    def _transform(self, X):
        if hasattr(self.clf, "predict_proba"):
            result = self.clf.predict_proba(X)
            if result.shape[1] == 2:
                result = result[:, 1]
        elif hasattr(self.clf, "predict"):
            result = self.clf.predict(X)
        else:
            result = self.clf.transform(X)
        return to2d(result)


class InfinityRemover(GenericTransform):

    """ Removes columns with infinite values.

        Looks for infinite values twice because the columns with infinite values in the training set may be different than in the test set. (fills these values with column means)
    """

    def _fit(self, X, y=None):
        self.no_inf = np.isinf(X).sum(axis=0) == 0

    def _transform(self, X):
        X_no_inf = X[:, self.no_inf]
        X_no_inf[np.isinf(X_no_inf)] = 0.0
        return X_no_inf


class NearZeroVarianceFilter(GenericTransform):

    """ Removes columns with standard deviation below a threshold
    """

    _default_args = dict(threshold=0.1)

    def _fit(self, X, y=None):
        self.keep_cols = X.std(axis=0) > self.threshold

    def _transform(self, X):
        return X[:, self.keep_cols]


class SparseFiltering(GenericTransform):

    """ Machine for calling sparse filtering algorithm.
    """

    _default_args = dict(N=1000)

    def _fit(self, X, y=None):
        self.W = sparse_filtering.sparseFiltering(self.N, X.T)

    def _transform(self, X):
        return sparse_filtering.feedForwardSF(self.W, X.T)


class RowApply(GenericTransform):

    """ apply function to each row
    """

    _default_args = dict(func=None)

    def _transform(self, X):
        return np.apply_along_axis(self.func, 1, X)


class ColApply(GenericTransform):

    """ apply function to each row
    """

    _default_args = dict(func=None)

    def _transform(self, X):
        return np.apply_along_axis(self.func, 0, X)


def row_normalizer(order=None):
    """ Normalize each row, given an order
    """
    return RowApply(func=partial(np.linalg.norm, ord=order))
