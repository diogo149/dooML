"""Table of Contents:
    -cv_fit_predict
    -multi_feature_scorer
    -machine_feature_scorer

"""

import numpy as np

from sklearn.cross_validation import StratifiedKFold, KFold

from utils import fit_predict, args_expander, kfold_feature_scorer, machine_score_func
from parallel import parmap, parfor


def cv_fit_predict(clf, X, y, stratified=False, n_folds=3, n_jobs=1):
    """ returns cross-validation predictions of a machine
    """
    assert isinstance(X, np.ndarray)
    kfold = list(StratifiedKFold(y, n_folds) if stratified else KFold(y.shape[0], n_folds, shuffle=True))
    items = [(clf, X[train_idx], y[train_idx], X[test_idx]) for train_idx, test_idx in kfold]
    mapped = parmap(args_expander, items, args=(fit_predict,), n_jobs=n_jobs)
    prediction = np.ones(y.shape)
    for (_, test_idx), vals in zip(kfold, mapped):
        prediction[test_idx] = vals
    return prediction


def multi_feature_scorer(num_features, score_func, n_iter=20, k=2, n_jobs=1):
    """ returns the average score for each feature over multiple iterations
    """
    mapped = parfor(kfold_feature_scorer, n_iter, args=(num_features, score_func, k), n_jobs=n_jobs)
    return np.mean(mapped, axis=0)


def machine_feature_scorer(clf, X, y, X_test, y_test, metric, n_iter=20, k=2, n_jobs=1):
    """ returns the score of each feature for a specific machine, dataset, and metric
    """

    def score_func(indices):
        return machine_score_func(clf, X[:, indices], y, X_test[:, indices], y_test, metric)

    num_features = X.shape[1]
    return multi_feature_scorer(num_features, score_func, n_iter, k, n_jobs)
