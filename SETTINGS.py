import numpy as np


class Setting(object):
    pass

MISC = Setting()

IS_CATEGORICAL = Setting()
IS_CATEGORICAL.THRESHOLD = 0.1

GAP_STATISTIC = Setting()
GAP_STATISTIC.RANDOMIZED_PCA_THRESHOLD = 10
GAP_STATISTIC.NUM_CLUSTERS_WITHOUT_IMPROVEMENT = 5
GAP_STATISTIC.MAXIMUM_DECLINE = 0.5

QUICK_CACHE = Setting()
QUICK_CACHE.DIRECTORY = "quick_cache"

DECORATORS = Setting()
DECORATORS.LOG = False
DECORATORS.MEMMAP_DTYPE = np.float32

UTILS = Setting()
UTILS.MEMMAP_DTYPE = np.float32

FEATURE_STORE = Setting()
FEATURE_STORE.DB = "feature.db"
FEATURE_STORE.DATA_NAME = "__TRAIN__"
FEATURE_STORE.CV_FOLDS = 16  # note that changing this in the middle of an experiment may make results less reproducible

PARALLEL = Setting()
PARALLEL.JOBS = -1
PARALLEL.JOBLIB_VERBOSE = 0
PARALLEL.JOBLIB_PRE_DISPATCH = 'n_jobs'
PARALLEL.PMAP = False

STORAGE = Setting()
STORAGE.JOBLIB_COMPRESSION = 9
STORAGE.COMPRESSION = 1
