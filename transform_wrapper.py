"""Table of Contents:
    -TransformWrapper
    -NoFit
    -QuickSave
    -RowSubset
    -ColSubset
    -Costing

    -Stateful !!!
"""
from __future__ import division
import numpy as np

from copy import deepcopy

from storage import quick_save
from utils import fit_predict, flexible_int_input
from utils2 import cv_fit_predict
from classes import GenericObject, Trial


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
        weights = np.ones(rows) if self.weights is None else self.weights / self.weights.max()
        weights = weights * weights.sum() * train_size / rows
        for i in xrange(self.n_iter):
            subset = weights > np.random.uniform(size=rows)
            trn = RowSubset(subset=subset, trn=deepcopy(self.trn))
            self.trns.append(trn.fit(X, y))
        return self

    def transform(self, X):
        return np.mean([trn.predict(X) for trn in self.trns], axis=0)


class Stateful(TransformWrapper):
    pass
#
#
# FIXME make sure this follows the transform assumptions


class Stateful_OLD(TransformWrapper):

    """ Creates cross-validated features when training.Requires being in a Trial in order to predict.
    """

    _default_args = dict(n_jobs=1, stratified=False, n_folds=3, validation_set=False, cv_X=None, cv_y=None)

    def predict(self, X, y=None):
        if Trial.train_mode():
            if self.validation_set:
                return fit_predict(self.trn, self.cv_X, self.cv_y, X)
            else:
                return cv_fit_predict(self.trn, X, y, self.stratified, self.n_folds, self.n_jobs)
        else:
            return self.trn.predict(X)

#
#
#
