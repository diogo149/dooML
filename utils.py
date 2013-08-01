"""Table of Contents:
    -debug
    -column_append
    -to_float
    -random_df
    -df_identifier
    -first_col
    -column_apply
    -combine_dfs
    -hash_numpy_int
    -hash_numpy
    -hash_df
    -smart_hash
    -binarize
    -current_time
    -print_current_time
    -random_seed
    -is_categorical
    -interaction_terms
    -add_index_to_columns
    -get_no_inf_cols
    -remove_inf_cols
    -args_expander
    -fit_predict
    -kfold_feature_scorer
    -machine_score_func
    -primes_to
    -is_prime

"""
from __future__ import print_function
import numpy as np
import pandas as pd
import random
import math

from pdb import set_trace
from datetime import datetime
from copy import deepcopy
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import KFold

import SETTINGS
from decorators import default_catcher

debug = set_trace


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


def hash_numpy_int(x):
    """Returns int of hashed value of numpy array.
    """
    assert isinstance(x, np.ndarray)
    return hash(tuple(map(to_float, x.flatten())))


def hash_numpy(x):
    """Returns string of hashed value of numpy array.
    """
    assert isinstance(x, np.ndarray)
    return "{}_{}".format(x.shape, hash_numpy_int(x))


def hash_df(df):
    """Returns hashed value of pandas data frame.
    """
    return hash_numpy(df.as_matrix())


def smart_hash(x):
    """ checks the type of x and hashes appropriately. returns a string hash
    """
    if x is None:
        return "None"
    elif isinstance(x, pd.DataFrame):
        return hash_df(x)
    elif isinstance(x, np.ndarray):
        return hash_numpy(x)
    elif isinstance(x, (int, str, float, bool, long)):
        return str(hash(x))
    else:
        raise Exception


def binarize(data):
    """ convert categorical data into a matrix of 0's and 1's
    """
    lb = LabelBinarizer()
    return lb.fit_transform(data)


def current_time():
    """ Returns current time as a string.
    """
    return str(datetime.now())


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


def get_no_inf_cols(x):
    """ returns which columns do not have infinite values
    """
    return np.isinf(x).sum(axis=0) == 0


def remove_inf_cols(no_inf, df):
    """
    Removes columns with infinite values.

    Looks for infinite values twice because the columns with infinite values in the training set may be different than in the test set. (fills these values with column means)
    """
    assert isinstance(df, pd.DataFrame)
    df_no_inf = df.ix[:, no_inf]
    df_no_inf[np.isinf(df_no_inf)] = 0
    return df_no_inf


def args_expander(func, item):
    """ takes in a function and an iterable, and expands the iterable as the functions arguments. useful for parallel maps.
    """
    return func(*item)


def fit_predict(clf, X, y, X_test):
    """ makes a copy of a machine, then fits it on training data and predicts on test data. useful for parallel maps.
    """
    tmp_clf = deepcopy(clf)
    tmp_clf.fit(X, y)
    return tmp_clf.predict(X_test)


def kfold_feature_scorer(num_features, score_func, k=2):
    """ returns the average scores for datasets with each feature, given a scoring function
    """
    result = np.zeros(num_features)
    for idx, _ in KFold(num_features, k, shuffle=True):
        result[idx] += score_func(idx)
    result /= (k - 1)
    return result


def machine_score_func(clf, X, y, X_test, y_test, metric):
    """ returns the score of a specific machine on a specific dataset with a specific metric
    """
    new_clf = deepcopy(clf)
    new_clf.fit(X, y)
    predictions = new_clf.predict(X_test)
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
