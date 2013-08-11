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

FEATURE_STORE = Setting()
FEATURE_STORE.DB = "feature.db"
FEATURE_STORE.DEBUG_SQL = False
FEATURE_STORE.DATA_NAME = "__TRAIN__"
