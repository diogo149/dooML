"""Table of Contents:
    -TransformWrapper
    -NoFit
    -QuickSave
    -RowSubset
    -ColSubset
    -RejectionSample
    -tuned_rejection_sample
"""
from __future__ import division
import numpy as np

from copy import deepcopy

from storage import quick_save

from utils import flexible_int_input, sample_tune
from classes import GenericObject


class TransformWrapper(GenericObject):

    """ parent class of a series of classes meant to wrap data transforms. there is no checking that a classifier is passed in
    """

    _required_args = ('trn',)

    def __getattr__(self, name):
        return getattr(self.trn, name)


class NoFit(TransformWrapper):

    """ Wraps a transform so that it's fit method doesn't change object state. Useful for models that are already fit.
    """

    def fit(self, X, y=None):
        return self


class QuickSave(TransformWrapper):

    """ Wraps a machine so that it saves a pickle of the classifier after training.
    """

    def fit(self, X, y=None):
        self.trn.fit(X, y)
        quick_save("quick_save_transform_wrapper", repr(self.trn), self.trn)
        return self


class RowSubset(TransformWrapper):

    """ trains a transform on only a subset of the training examples, defined by the input subset
    """

    _required_args = ('subset', 'trn',)

    def fit(self, X, y=None):
        X = X[self.subset, :]
        if y is not None:
            y = y[self.subset, :]
        return self.trn.fit(X, y)


class ColSubset(TransformWrapper):

    """ trains a transform on only a subset of the features, defined by the input subset
    """

    _required_args = ('subset', 'trn',)

    def fit(self, X, y=None):
        X = X[:, self.subset]
        return self.trn.fit(X, y)


class RejectionSample(TransformWrapper):

    """ uses rejection sampling with input weights
    """

    _default_args = dict(weights=None, train_size=None, n_iter=100)

    def fit(self, X, y=None):
        self.trns = []
        rows = X.shape[0]
        train_size = flexible_int_input(self.train_size, rows)
        if self.weights is None:
            weights = np.ones(rows) * train_size / rows
        else:
            weights = np.abs(self.weights.flatten())
            weights = weights / weights.max()
            if weights.sum() >= train_size:
                weights = weights * train_size / weights.sum()
            else:
                weights = 1 - (1 - weights) * (rows - train_size) / (rows - weights.sum())
        # for i in xrange(self.n_iter):
        #     subset = weights > np.random.uniform(size=rows)
        #     trn = RowSubset(subset=subset, trn=deepcopy(self.trn))
        #     self.trns.append(trn.fit(X, y))
        subsets = (weights > np.random.uniform(size=rows) for _ in xrange(self.n_iter))
        self.trns = pmap(self.pmap_helper, subsets, X=X, y=y)
        return self

    def pmap_helper(self, subset, X, y):
        trn = RowSubset(subset=subset, trn=deepcopy(self.trn))
        trn.fit(X, y)
        return trn

    def transform(self, X):
        return np.mean([trn.transform(X) for trn in self.trns], axis=0)


def tuned_rejection_sample(trn, X_col, weights=None, n_iter=100, y_col=1, y_categories=1, seconds=10):
    """ returns a RejectionSample wrapped transform with the quantity tuned to take a certain amount of time
    """
    train_size = sample_tune(trn, X_col, y_col, y_categories, seconds)
    return RejectionSample(trn=trn, weights=weights, train_size=train_size, n_iter=n_iter)
