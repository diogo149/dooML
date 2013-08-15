"""Table of Contents:
    -cv_fit_transform
    -multi_feature_scorer
    -machine_feature_scorer

"""

import numpy as np

from sklearn.cross_validation import StratifiedKFold, KFold

from utils import fit_transform, args_expander, kfold_feature_scorer, machine_score_func
from parallel import parfor, pmap


def cv_fit_transform(trn, X, y=None, stratified=False, n_folds=3):
    """ return cross-validated transform features
    """
    rows = X.shape[0]
    n_folds = min(n_folds, rows)
    kfold = list(StratifiedKFold(y, n_folds) if stratified else KFold(rows, n_folds, shuffle=True))
    items = ((trn, X[train_idx], y[train_idx], X[test_idx]) for train_idx, test_idx in kfold)
    mapped = pmap(args_expander, items, fit_transform)
    transformed = [None] * rows
    for (_, test_idx), vals in zip(kfold, mapped):
        for idx, val in zip(test_idx, vals):
            transformed[idx] = val
    return np.array(transformed)


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

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from transform import SklearnBridge
    X = np.random.randn(100, 100)
    y = np.random.randint(0, 2, size=(100, 1))
    trn = SklearnBridge(clf=LogisticRegression())
    p = cv_fit_transform(trn, X, y)
    print (((p[:, 1] > 0.5) + 0).reshape(-1, 1) == y).sum()
