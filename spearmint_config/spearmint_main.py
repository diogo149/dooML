import numpy as np
import cPickle as pickle
import gc
from os import path, makedirs
from math import exp


def try_mkdir(directory):
    """ try to make directory
    """
    try:
        makedirs(directory)
    except OSError:
        pass


def quick_save(directory, filename, obj):
    """Quickly pickle an object in a file.
    """
    try_mkdir(directory)
    gc.disable()
    new_filename = path.join(directory, filename + ".pickle")
    with open(new_filename, 'w') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
    gc.enable()


def quick_load(directory, filename):
    """Quickly unpickle an object from a file.
    """
    new_filename = path.join(directory, filename + ".pickle")
    gc.disable()
    with open(new_filename) as infile:
        obj = pickle.load(infile)
    gc.enable()
    return obj


def process_params(params):
    for key in params.keys():
        if len(params[key]) == 1:
            params[key] = params[key][0]
            if key.startswith('log_'):
                params[key[4:]] = exp(params[key])
                params.pop(key)
    return params


def call_external(params):
    import re
    from subprocess import check_output
    data = check_output(['octave', '-qf', 'run.m'] + params)
    result = re.search(r'([0-9.]+)', data).group(0)
    return result


def sample_func(params):
    from sklearn.linear_model import SGDClassifier
    from sklearn.cross_validation import StratifiedKFold, KFold
    clf = SGDClassifier(penalty='elasticnet', n_iter=100, shuffle=True, **params)
    X = quick_load(".", "train_data")
    y = quick_load(".", "ans")
    for train, test in StratifiedKFold(y, n_folds=10):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        predictions = clf.fit(X_train, y_train).predict(X_test)
        return np.abs(predictions - y_test).sum() / len(y_test)


def main(job_id, params):
    # this will land in output/ files
    print 'Job id: ', str(job_id)
    print params
    np.random.seed(int(job_id))
    from somewhere import somefunc as func
    result = func(process_params(params))
    print 'result: ', result
    return result
