from storage import quick_load
from utils import bitmask_score_func
from helper.bitmasks import bitmask_genetic_algorithm

if __name__ == "__main__":
    X = quick_load(".", "???")
    y = quick_load(".", "???")
    score_func = "???"
    clf = None
    if 0:  # using a separate validation set
        X_test = quick_load(".", "???")
        y_test = quick_load(".", "???")
        sf = bitmask_score_func(clf, X, y, score_func, cv=False, X_test=X_test, y_test=y_test, cache=False)
    else:  # using CV
        sf = bitmask_score_func(clf, X, y, score_func, stratified=False, n_folds=10, checked_folds=1)

    gene_pool = bitmask_genetic_algorithm(X.shape[1], sf, epochs=100, population_size=32, avg_mutations=100, child_rate=0.5, n_jobs=1)
