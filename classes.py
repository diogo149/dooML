"""Table of Contents
    -GenericObject
    -ContextManagerObject
    -INSTANCE_GETATTR
    -SingletonContextManagerObject
    -Trial
    -FeatureCache
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
    -MachineFactory
    -infinity_remover
    -ValidationFeature
    -ResidualPredicter

    -BinningMachine
    -NumpyBinningMachine
    -classification_sampling_machine
    -regression_sampling_machine
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import re
import random
import warnings

from time import time
from copy import deepcopy
from types import MethodType
from functools import partial
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

from storage import quick_write, quick_save, quick_load, quick_exists
from utils import is_categorical, hash_df, get_no_inf_cols, remove_inf_cols, fit_predict, cv_fit_predict
from helper import sparse_filtering, gap_statistic

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
            trained_clf = self.clfs.pop(0)
            assert isinstance(trained_clf, clf.__class__)
            return trained_clf

    def log(self, **kwargs):
        self(**kwargs)

    def train_mode(self):
        return self.train_mode


class FeatureCache(object):

    def __init__(self, df):
        self._rows = df.shape[0]
        self._directory = hash_df(df)
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


class MachineFactory(GenericObject):

    """ Class that allows wrapping of functions into a predicting machine. Assumes that the first input to the prediction function is the output of the fit function.
    """

    _required_args = ('fit_func', 'predict_func')

    def fit(self, *args, **kwargs):
        self.fit_params = self.fit_func(*args, **kwargs)
        return self

    def predict(self, *args, **kwargs):
        return self.predict_func(self.fit_params, *args, **kwargs)


def infinity_remover():
    """ returns a class that removes infinity columns from a data frame
    """
    return MachineFactory(fit_func=get_no_inf_cols, predict_func=remove_inf_cols)


class ValidationFeature(MachineWrapper):

    """ Creates cross-validated features when training.Requires being in a Trial in order to predict.
    """

    _default_args = dict(n_jobs=1, stratified=False, n_folds=3, validation_set=False, cv_X=None, cv_y=None)

    def predict(self, X, y=None):
        if Trial.train_mode():
            if self.validation_set:
                return fit_predict(self.clf, self.cv_X, self.cv_y, X)
            else:
                return cv_fit_predict(self.clf, X, y, self.stratified, self.n_folds, self.n_jobs)
        else:
            return self.clf.predict(X)

    def fit_predict(self, item):
        X, y, X_test = item
        tmp_clf = deepcopy(self.clf)
        tmp_clf.fit(X, y)
        return tmp_clf.predict(X_test)


class ResidualPredicter(GenericObject):

    """ Predicts the residual of a classifier using another classifier.
    """

    _required_args = ('clf1',)
    _default_args = dict(clf2=None, n_folds=3, n_jobs=1)

    # get validation feature

    def fit(self, X, y):
        self.clf1.fit(X, y)
        predictions = cv_fit_predict(self.clf1, X, y, n_folds=self.n_folds, n_jobs=self.n_jobs)
        if self.clf2 is None:
            self.clf2 = GradientBoostingRegressor(loss='huber', n_estimators=100, subsample=0.5, max_features=int(np.sqrt(X.shape[1])))
        self.clf2.fit(X, predictions)
        return y - self.clf2.predict(X)

    def predict(self, *args, **kwargs):
        return self.clf2.predict(*args, **kwargs)


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
