from utils import spearmint_params, random_seed, quick_cv, fit_predict
from storage import quick_load


def score_func(y_true, y_pred):
    return 0


def main_func(params):
    clf = None
    X = quick_load(".", "???")
    y = quick_load(".", "???")
    if 0:  # using a separate validation set
        X_test = quick_load(".", "???")
        y_test = quick_load(".", "???")
        return score_func(y_test, fit_predict(X, y, X_test, cache=True))
    else:  # using CV
        return quick_cv(clf, X, y, score_func, n_folds=10, check_folds=1)


def main(job_id, params):
    print 'Job id: ', str(job_id)  # this will land in output/ files
    print params
    random_seed(int(job_id))
    result = main_func(spearmint_params(params))
    print 'result: ', result
    return result
