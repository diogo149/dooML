"""Table of Contents:
    -debug
    -column_append
    -to_float
    -random_df
    -df_identifier
    -first_col
    -column_apply
    -combine_dfs
    -smart_hash
    -binarize
    -current_time
    -print_current_time
    -random_seed
    -is_categorical
    -interaction_terms
    -add_index_to_columns
    -args_expander
    -fit_predict
    -fit_transform
    -kfold_feature_scorer
    -machine_score_func
    -primes_to
    -is_prime
    -extract_output
    -spearmint_params
    -quick_cv
    -grouper
    -quick_score
    -feature_bitmask
    -bitmask_score_func
    -flexible_int_input
    -to2d
    -any_iterable
    -all_iterable
    -all_isinstance
    -bool_to_int
    -to_memmap
    -memmap_hstack
    -set_debug_logging
    -sample_tune

"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import joblib
import random
import re
import itertools
import tempfile
import time
import logging
import math
import subprocess
import datetime
import copy
import pdb

from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import KFold, StratifiedKFold

import SETTINGS
from decorators import default_catcher, log
from storage import machine_cache

debug = pdb.set_trace


def column_append(s, df):
    """
    Appends a value to the name of a dataframe.
    """
    assert isinstance(df, pd.DataFrame)
    return df.rename(columns=lambda c: "{}_{}".format(c, s))


@default_catcher(np.nan)
def to_float(s):
    return float(s)


def random_df(rows, cols):
    """
    Returns a normally distributed random dataframe with input dimensions.
    """
    return pd.DataFrame(np.random.randn(rows, cols))


def df_identifier(df):
    """
    Gets a (hopefully) unique identifier for a dataframe.
    """
    assert isinstance(df, pd.DataFrame)
    return "".join(map(str, df.columns))


def first_col(df):
    """
    Returns the first column of a DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    return df[df.columns[0]]


def column_apply(df, func, *args, **kwargs):
    """
    Returns the resulting dataframe from applying a function to each column of an input dataframe.
    """
    assert isinstance(df, pd.DataFrame)
    if df.shape[1] == 1:
        return func(df)
    elif df.shape[1] == 0:
        return pd.DataFrame()
    applied = [func(df[[col]], *args, **kwargs) for col in df]
    return combine_dfs(applied)


def combine_dfs(dfs):
    """
    Takes in a list of dataframes with the same number of rows and appends them together.
    """
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, axis=1)


def smart_hash(x):
    """ returns a string hash
    """
    return joblib.hash(x)


def binarize(data):
    """ convert categorical data into a matrix of 0's and 1's
    """
    lb = LabelBinarizer()
    return lb.fit_transform(data)


def current_time():
    """ Returns current time as a string.
    """
    return str(datetime.datetime.now())


def print_current_time():
    """ prints current time
    """
    print(current_time())


def random_seed(x):
    """ seeds random state
    """
    assert isinstance(x, int)
    assert x >= 0
    random.seed(x)
    np.random.seed(x)


def is_categorical(X):
    """ utility method to determine whether a feature is categorical
    """
    assert isinstance(X, np.ndarray)
    size, = X.shape
    try:  # non-numerical
        X = X.astype(np.float)
    except ValueError:
        return True
    if not np.allclose(X, X.astype(np.int)):  # floating point numbers
        return False
    num_unique, = np.unique(X).shape
    if num_unique > SETTINGS.IS_CATEGORICAL.THRESHOLD * size:
        return False
    else:
        return True


def interaction_terms(df, feat1, feat2):
    """Creates a new df with (len(feat1) + 1) * (len(feat2) + 1) - 1 features, with each new feature as a product of a feature in feat1 and a feature in feat2.
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(feat1, set)
    assert isinstance(feat2, set)
    assert feat1.intersection(feat2) == set()
    cols = set(df.columns)
    assert feat1.issubset(cols)
    assert feat2.issubset(cols)

    terms = []
    feat1_df = df[list(feat1)]
    terms.append(feat1_df)
    terms.append(df[list(feat2)])

    for f in feat2:
        col = np.array(df[f]).reshape(-1, 1)
        prod = col * feat1_df
        renamed = prod.rename(columns=lambda x: "{}_*_{}".format(x, f))
        terms.append(renamed)

    return combine_dfs(terms)


def add_index_to_columns(df):
    """ adds an index number to each column of a dataframe (useful if two or more columns share a name)
    """
    return pd.DataFrame(df.as_matrix(), columns=["{}_{}".format(i, j) for i, j in enumerate(df.columns)])


def args_expander(func, item):
    """ takes in a function and an iterable, and expands the iterable as the functions arguments. useful for parallel maps.
    """
    return func(*item)


def fit_predict(clf, X, y, X_test, cache=False):
    """ makes a copy of a machine, then fits it on training data and predicts on test data. useful for parallel maps.
    """
    tmp_clf = copy.deepcopy(clf)
    if cache:
        tmp_clf = machine_cache(smart_hash(X), tmp_clf, X, y)
    else:
        tmp_clf.fit(X, y)
    return tmp_clf.predict(X_test)


def fit_transform(trn, X, y, X_test):
    """ makes a copy of a transform, then fits it on training data and predicts on test data. useful for parallel maps.
    """
    temp_trn = copy.deepcopy(trn)
    temp_trn.fit(X, y)
    return temp_trn.transform(X_test)


def kfold_feature_scorer(num_features, score_func, k=2):
    """ returns the average scores for datasets with each feature, given a scoring function
    """
    result = np.zeros(num_features)
    for idx, _ in KFold(num_features, k, shuffle=True):
        result[idx] += score_func(idx)
    result /= (k - 1)
    return result


def machine_score_func(clf, X, y, X_test, y_test, metric, cache=False):
    """ returns the score of a specific machine on a specific dataset with a specific metric
    """
    predictions = fit_predict(clf, X, y, X_test, cache=cache)
    return metric(y_test, predictions)


def primes_to(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    if n < 6:
        return [i for i in [2, 3, 5] if i <= n]
    sieve = np.ones(n / 3 + (n % 6 == 2), dtype=np.bool)
    sieve[0] = False
    for i in xrange(int(n ** 0.5) / 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[((k * k) / 3)::2 * k] = False
            sieve[(k * k + 4 * k - 2 * k * (i & 1)) / 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0] + 1) | 1)]


def is_prime(n):
    """ returns True if N is a prime number, False otherwise
    """
    if n != int(n) or n <= 1:
        return False
    upper_bound = int(np.sqrt(n))
    for p in primes_to(upper_bound):
        if n % p == 0:
            return False
    return True


def extract_output(cmd_list, pattern):
    """ performs a search for a pattern in the stdout of a command and returns a list of all matches
    """
    output = subprocess.check_output(cmd_list)
    return re.findall(pattern, output)


def spearmint_params(params):
    """ process input params from spearmint by extracting elements from one element lists and transforming logged input
    """
    for key in params.keys():
        if len(params[key]) == 1:
            params[key] = params[key][0]
            if key.startswith('log_'):
                params[key[4:]] = math.exp(params[key])
                params.pop(key)
    return params


def quick_cv(clf, X, y, score_func, stratified=False, n_folds=10, checked_folds=1):
    """ returns the score of one (or more) fold(s) of cross validation, for quickly getting a score
    """
    if stratified:
        cv = StratifiedKFold(y, n_folds=n_folds)
    else:
        cv = KFold(y.shape[0], n_folds=n_folds, shuffle=True)
    scores = []
    for train, test in cv:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(score_func(y_test, fit_predict(clf, X_train, y_train, X_test)))
        if len(scores) >= checked_folds:
            break
    return sum(scores) / float(len(scores))


def grouper(n, iterable, fillvalue=None):
    """ groups an iterable into chunks of a certain size
    >>> grouper(3, 'ABCDEFG', 'x')
    ABC DEF Gxx

    from: http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
    """
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def quick_score(clf, X, y, score_func, cv=True, X_test=None, y_test=None, cache=False, stratified=False, n_folds=10, checked_folds=1):
    """ all in one scoring function that can handle both cross validation and an external validation set
    """
    if cv:
        return quick_cv(clf, X, y, score_func=score_func, stratified=stratified, n_folds=n_folds, checked_folds=checked_folds)
    else:
        return score_func(y_test, fit_predict(clf, X, y, X_test, cache=cache))


def feature_bitmask(X, bitmask):
    """ keeps only columns of a 2D numpy array corresponding to the input bitmask
    """
    return X[:, np.where(bitmask)[0]]


def bitmask_score_func(clf, X, y, score_func, cv=True, X_test=None, y_test=None, cache=False, stratified=False, n_folds=10, checked_folds=1):
    """ returns a function that takes in a bitmask and returns it's respective score
    """
    def wrapped(bitmask):
        new_X = feature_bitmask(X, bitmask)
        if X_test is not None:
            new_X_test = feature_bitmask(X_test, bitmask)
        else:
            new_X_test = None
        return quick_score(clf=clf, X=new_X, y=y, score_func=score_func, cv=cv, X_test=new_X_test, y_test=y_test, cache=cache, stratified=stratified, n_folds=n_folds, checked_folds=checked_folds)
    return wrapped


def flexible_int_input(in_val, size):
    """ allows for flexible input as a size
    """
    if in_val is None:
        return size
    elif isinstance(in_val, float) and 0 < in_val <= 1.0:
        return int(round(in_val * size))
    elif isinstance(in_val, int) and in_val > 1:
        return min(size, in_val)
    elif in_val == "sqrt":
        return int(round(math.sqrt(size)))
    elif in_val == "log2":
        return int(round(math.log(size) / math.log(2)))
    elif in_val == "auto":
        return size


def to2d(x):
    """ returns a two-dimensional array
    """
    if len(x.shape) == 1:
        return x.reshape(-1, 1)
    elif len(x.shape) == 2:
        return x
    else:
        raise Exception


def any_iterable(iterable, func):
    """ checks if function is true for any item in iterable
    """
    return any(func(item) for item in iterable)


def all_iterable(iterable, func):
    """ checks if function is true for all items in iterable
    """
    return all(func(item) for item in iterable)


def all_isinstance(iterable, t):
    """ check if each object in an iterable is an instance of a type
    """
    return all_iterable(iterable, lambda x: isinstance(x, t))


def bool_to_int(boolean):
    """ integer equivalent to boolean
    """
    return 1 if boolean else 0


def to_memmap(X):
    """ returns a memmap of the input numpy array
    """
    dtype = X.dtype if SETTINGS.UTILS.MEMMAP_DTYPE is None else SETTINGS.UTILS.MEMMAP_DTYPE
    with tempfile.TemporaryFile() as infile:
        memmapped = np.memmap(infile, dtype=dtype, shape=X.shape)
    memmapped[:] = X
    return memmapped


def memmap_hstack(Xs):
    """ returns a memmapped hstack of input arrays in a memory efficient manner
    """
    assert len(Xs)
    shapes = [X.shape for X in Xs]
    assert all_iterable(shapes, lambda x: len(x) == 2), "input array is not 2D"
    columns, rows = zip(*shapes)
    assert min(columns) == max(columns), "input does not have same number of columns"

    total_columns = columns[0]
    total_rows = sum(rows)
    dtype = np.float if SETTINGS.UTILS.MEMMAP_DTYPE is None else SETTINGS.UTILS.MEMMAP_DTYPE
    with tempfile.TemporaryFile() as infile:
        memmapped = np.memmap(infile, dtype=dtype, shape=(total_columns, total_rows))

    offset = 0
    for X in Xs:
        row = X.shape[0]
        memmapped[:, offset:offset + row] = X
        offset += row
    return memmapped


def set_debug_logging(log=True):
    """ activates or deactivates logging debug statements
    """
    if log:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def sample_tune(trn, X_col, y_col=1, y_categories=1, seconds=10):
    """ determines the ideal amount of input data (rows) to train a transform in a certain amount of time assuming that fitting the transform is the bottleneck
    -assumes that data is normally distributed
    -won't be accurate for convergence based methods
    -dtype used will be np.float32
    -assuming time = a * n ** p -> p = log_n (time) - log_n(a) - lower order terms
    """
    @log
    def fit_time(num):
        num = int(round(num))
        X = np.random.randn(num, X_col).astype(np.float32)
        y = np.random.randn(num, y_col) if y_categories < 2 else np.random.randint(0, y_categories, (num, y_col))
        y = y.astype(np.float32)
        start_time = time.time()
        trn.fit(X, y)
        return time.time() - start_time

    num, factor, tol = 20, 2.0, 0.1
    prev = fit_time(num / factor)
    for _ in xrange(100):
        next = fit_time(num)
        growth = math.log(next / prev) / math.log(factor)
        growth = max(growth, 2)
        factor = (seconds / next) ** (1.0 / growth)
        prev = next
        num *= factor
        if abs(1 - seconds / next) <= tol:
            break

    return num
