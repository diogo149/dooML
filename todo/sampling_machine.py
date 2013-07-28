import numpy as np
import multiprocessing

from copy import deepcopy
from scipy import stats
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit


def mode_wrapper(*args, **kwargs):
    return stats.mode(*args, **kwargs)[0]


class SamplingMachine(object):

    COMBINE_FUNCS = {
        "mean": np.mean,
        "mode": mode_wrapper,
        "median": np.median,
    }

    def __init__(self, clf, predict_combine="mean", sample_size=200, stratified_shuffle=False, iterations=10, n_jobs=1, random_state=1):
        assert predict_combine in SamplingMachine.COMBINE_FUNCS
        assert isinstance(n_jobs, int)
        self.clf = deepcopy(clf)
        self.predict_combine = predict_combine
        self.sample_size = sample_size
        self.stratified_shuffle = stratified_shuffle
        self.iterations = iterations
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.random_state = random_state

    def fit(self, X, y):
        train = np.array(X)
        assert len(train.shape) == 2
        assert len(y.shape) == 1
        self.shuffle_split_indices(y)
        n_rows, _ = train.shape
        self.clfs = []
        for indices in self.shuffle_split_indices(y):
            tmp_clf = deepcopy(self.clf)
            tmp_clf.fit(train[indices], y[indices])
            self.clfs.append(tmp_clf)

    def predict(self, X):
        data = []
        for clf in self.clfs:
            data.append(clf.predict(X))
        return SamplingMachine.combine(data, self.predict_combine)

    def predict_proba(self, X):
        data = []
        for clf in self.clfs:
            data.append(clf.predict_proba(X))
        return SamplingMachine.combine(data)

    def shuffle_split_indices(self, y):
        n_rows = len(y)
        params = dict(n_iter=self.iterations, test_size=self.sample_size, random_state=self.random_state)
        if n_rows <= self.sample_size:
            params['n_iter'] = 1
            params['test_size'] = n_rows
        if self.stratified_shuffle:
            shuffle_split = StratifiedShuffleSplit(y, **params)
        else:
            shuffle_split = ShuffleSplit(n_rows, **params)
        return [index for _, index in shuffle_split]

    @staticmethod
    def combine(data, predict_combine="mean"):
        combine_func = SamplingMachine.COMBINE_FUNCS[predict_combine]
        return combine_func(np.array(data), axis=0)


def classification_sampling_machine(clf, predict_combine="mode", sample_size=200, stratified_shuffle=True, iterations=10, n_jobs=1, random_state=1):
    return SamplingMachine(clf, predict_combine, sample_size, stratified_shuffle, iterations, n_jobs, random_state)


def regression_sampling_machine(clf, predict_combine="mean", sample_size=200, stratified_shuffle=False, iterations=10, n_jobs=1, random_state=1):
    return SamplingMachine(clf, predict_combine, sample_size, stratified_shuffle, iterations, n_jobs, random_state)

if __name__ == "__main__":
    from sklearn.svm import LinearSVC, SVR
    x = np.random.randn(1000, 1000)
    z = np.random.randn(1000)
    z = x.sum(axis=1) ** 2
    z -= z.mean()
    z /= z.std()
    y = (z > 0) + 0
    raise Exception
    clf = LinearSVC()
    sm = classification_sampling_machine(clf)
    sm.fit(x, y)  # 0.40s
    print (sm.predict(x) == y).sum()  # 689
    clf.fit(x, y)  # 0.94
    print (clf.predict(x) == y).sum()  # 1000

    clf2 = SVR(kernel='linear')
    sm2 = regression_sampling_machine(clf)
    sm2.fit(x, z)  # 54.53s
    print ((sm2.predict(x) - z) ** 2).mean()  # 0.787013032134
    clf2.fit(x, z)  # 587.59 s
    print ((clf2.predict(x) - z) ** 2).mean()  # 0.0221129567591

    # %time clf = SVR(kernel='linear'); sm = regression_sampling_machine(clf); sm.fit(x, z); print ((sm.predict(x) - z) ** 2).mean()
    # 0.698017772415
