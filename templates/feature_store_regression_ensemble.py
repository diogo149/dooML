import numpy as np
import multiprocessing

import SETTINGS
from utils import flexible_int_input, set_debug_logging
from transform import sklearn_bridge, SparseFiltering, feast_bridge
from transform_wrapper import tuned_rejection_sample
from feature_store import FeatureStore, NodeFactory, FeatureDependency

""" clean up """
import os
os.system('rm -f feature.db')
os.system('rm -rf __TRAIN__')

""" #INPUT """
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=500, n_informative=250)
TRANSFORM_TIME = 1.0e-1  # seconds
N_ITER = 1  # times sampling occurs for each *_TO_ONE transform

""" Setting variables """
set_debug_logging()
y = y.reshape(-1, 1)
BIG_COLUMNS = X.shape[1]
SQRT_COLUMNS = flexible_int_input("sqrt", BIG_COLUMNS)
SETTINGS.FEATURE_STORE.CV_FOLDS = multiprocessing.cpu_count()

store = FeatureStore()
store.input_node(data=X, name="X", tags=("X", "input"))
store.input_node(data=y, name="y", tags=("y", "input"))
y_dep = FeatureDependency(parent_labels=["y"])
factory = NodeFactory(store)

"""
BIG_TO_SQRT
    -dimensionality reduction
    -feature selection
BIG_TO_ONE
    -prediction
SQRT_TO_SQRT
    -feature extraction
SQRT_TO_ONE
    -prediction

total # of ONE features should be:
    |BIG_TO_ONE| + |BIG_TO_SQRT| * (|SQRT_TO_SQRT| + 1) * |SQRT_TO_ONE|
"""

BIG_TO_ONE = []
SQRT_TO_ONE = []
BIG_TO_SQRT = []
SQRT_TO_SQRT = []

""" tree methods """
for max_depth in [2, 3, 4, None]:
    for max_features in ["auto", "sqrt", "log2"]:
        BIG_TO_ONE.append(sklearn_bridge().ensemble.RandomForestRegressor(max_features=max_features, max_depth=max_depth))
        BIG_TO_ONE.append(sklearn_bridge().ensemble.ExtraTreesRegressor(max_features=max_features, max_depth=max_depth))
        SQRT_TO_ONE.append(sklearn_bridge().ensemble.RandomForestRegressor(max_features=max_features, max_depth=max_depth))
        SQRT_TO_ONE.append(sklearn_bridge().ensemble.ExtraTreesRegressor(max_features=max_features, max_depth=max_depth))
    for loss in ['ls', 'lad', 'huber', 'quantile']:
        BIG_TO_ONE.append(sklearn_bridge().ensemble.GradientBoostingRegressor(loss=loss, max_depth=max_depth))
        SQRT_TO_ONE.append(sklearn_bridge().ensemble.GradientBoostingRegressor(loss=loss, max_depth=max_depth))
for loss in ['linear', 'square', 'exponential']:
    BIG_TO_ONE.append(sklearn_bridge().ensemble.AdaBoostRegressor(loss=loss))
    SQRT_TO_ONE.append(sklearn_bridge().ensemble.AdaBoostRegressor(loss=loss))

# can't use generalized_exponential: Exception: Length of theta must be 2 or 501
""" gaussian processes """
for corr in ['absolute_exponential', 'squared_exponential', 'cubic', 'linear']:
    for nugget in [1e-14, 1e-10, 1e-5, 1e-1]:
        BIG_TO_ONE.append(sklearn_bridge().gaussian_process.GaussianProcess(corr=corr, nugget=nugget))
        SQRT_TO_ONE.append(sklearn_bridge().gaussian_process.GaussianProcess(corr=corr, nugget=nugget))

""" k-nearest neighbor models """
for weights in ["uniform", "distance"]:
    for n_neighbors in [1, 5, 20]:
        BIG_TO_ONE.append(sklearn_bridge().neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))
        SQRT_TO_ONE.append(sklearn_bridge().neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))

""" cross_decomposition """
for n_components in [2, 3, 10]:
    BIG_TO_ONE.append(sklearn_bridge().cross_decomposition.PLSRegression(n_components=n_components))
    SQRT_TO_ONE.append(sklearn_bridge().cross_decomposition.PLSRegression(n_components=n_components))

""" support vector machines """
for C in [0.001, 0.01, 0.1, 1.0, 10]:
    BIG_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='linear', C=C))
    SQRT_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='linear', C=C))
    BIG_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='sigmoid', C=C))
    SQRT_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='sigmoid', C=C))
    for gamma in [0.0, 0.001, 0.01, 0.1]:
        BIG_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='rbf', C=C, gamma=gamma))
        SQRT_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='rbf', C=C, gamma=gamma))
        BIG_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='poly', C=C, gamma=gamma))
        SQRT_TO_ONE.append(sklearn_bridge().svm.SVR(kernel='poly', C=C, gamma=gamma))

""" linear models """
BIG_TO_ONE.append(sklearn_bridge().linear_model.ARDRegression())
SQRT_TO_ONE.append(sklearn_bridge().linear_model.ARDRegression())
BIG_TO_ONE.append(sklearn_bridge().linear_model.BayesianRidge())
SQRT_TO_ONE.append(sklearn_bridge().linear_model.BayesianRidge())
for alpha in [0.001, 0.01, 0.1, 1.0, 10]:
    BIG_TO_ONE.append(sklearn_bridge().linear_model.Ridge(alpha=alpha))
    SQRT_TO_ONE.append(sklearn_bridge().linear_model.Ridge(alpha=alpha))

BIG_TO_SQRT.append(sklearn_bridge().decomposition.RandomizedPCA(n_components=SQRT_COLUMNS))
BIG_TO_SQRT.append(sklearn_bridge().decomposition.FactorAnalysis(n_components=SQRT_COLUMNS))
BIG_TO_SQRT.append(sklearn_bridge().decomposition.FastICA(n_components=SQRT_COLUMNS))
BIG_TO_SQRT.append(feast_bridge().JMI(n_select=SQRT_COLUMNS))
BIG_TO_SQRT.append(feast_bridge().DISR(n_select=SQRT_COLUMNS))
BIG_TO_SQRT.append(feast_bridge().CMIM(n_select=SQRT_COLUMNS))
BIG_TO_SQRT.append(feast_bridge().MIM(n_select=SQRT_COLUMNS))
BIG_TO_SQRT.append(feast_bridge().mRMR(n_select=SQRT_COLUMNS))

SQRT_TO_SQRT.append(sklearn_bridge().neural_network.BernoulliRBM(n_components=SQRT_COLUMNS))
# removing GMM because it doesn't take in a y for fit
# SQRT_TO_SQRT.append(sklearn_bridge().mixture.GMM(n_components=SQRT_COLUMNS))
SQRT_TO_SQRT.append(sklearn_bridge().cluster.SpectralClustering(n_clusters=SQRT_COLUMNS))
SQRT_TO_SQRT.append(sklearn_bridge().cluster.Ward(n_clusters=SQRT_COLUMNS))
SQRT_TO_SQRT.append(sklearn_bridge().kernel_approximation.Nystroem(n_components=SQRT_COLUMNS))
SQRT_TO_SQRT.append(sklearn_bridge().kernel_approximation.RBFSampler(n_components=SQRT_COLUMNS))
SQRT_TO_SQRT.append(SparseFiltering(n_components=SQRT_COLUMNS))

"""
note to self, make sure to set n_iter=1 for transforms

functions:
NodeFactory for (take in rows and seconds as arguments to auto-tune)
        sklearn
        node
    tag_dependency function
    label_dependency function

"""
scaler_trn = sklearn_bridge().preprocessing.StandardScaler()
factory.foreach("X", scaler_trn, label=None, new_tags=["BIG"], y=y_dep)

# popping to avoid memory leaks
while BIG_TO_ONE:
    big2one = BIG_TO_ONE.pop()
    try:
        trn = tuned_rejection_sample(big2one, BIG_COLUMNS, n_iter=N_ITER, seconds=TRANSFORM_TIME)
    except:
        print("Tuning Failed: {}".format(big2one))
        continue
    factory.foreach("BIG", trn, label=None, new_tags=["ONE", "BIG2ONE"], y=y_dep)
while BIG_TO_SQRT:
    big2sqrt = BIG_TO_SQRT.pop()
    try:
        trn = tuned_rejection_sample(big2sqrt, BIG_COLUMNS, n_iter=1, seconds=TRANSFORM_TIME)
    except:
        print("Tuning Failed: {}".format(big2sqrt))
        continue
    factory.foreach("BIG", trn, label=None, new_tags=["SQRT", "BIG2SQRT"], y=y_dep)

while SQRT_TO_SQRT:
    sqrt2sqrt = SQRT_TO_SQRT.pop()
    try:
        trn = tuned_rejection_sample(sqrt2sqrt, SQRT_COLUMNS, n_iter=1, seconds=TRANSFORM_TIME)
    except:
        print("Tuning Failed: {}".format(sqrt2sqrt))
        continue
    factory.foreach("BIG2SQRT", trn, label=None, new_tags=["SQRT", "SQRT2SQRT"], y=y_dep)

while SQRT_TO_ONE:
    sqrt2one = SQRT_TO_ONE.pop()
    try:
        trn = tuned_rejection_sample(sqrt2one, SQRT_COLUMNS, n_iter=N_ITER, seconds=TRANSFORM_TIME)
    except:
        print("Tuning Failed: {}".format(sqrt2one))
        continue
    factory.foreach("SQRT", trn, label=None, new_tags=["ONE", "SQRT2ONE"], y=y_dep)

trn = tuned_rejection_sample(sklearn_bridge().linear_model.SGDRegressor(loss="huber"), 20000, n_iter=10 * N_ITER, seconds=10 * TRANSFORM_TIME)
ensemble = factory.forall("ONE", trn, label="ensemble", new_tags=["ENSEMBLE"], y=y_dep)

if 0:
    store = FeatureStore()
    ensemble = store.node(store.node_id(("ensemble", -1)))
    output = store.fit_transform(ensemble)
    store.input_node(data=X, name="X", tags=("X", "input"))
    store.input_node(data=y, name="y", tags=("y", "input"))
    X_node = store.node(store.node_id(("X", -1)))
    # y_node = store.node(store.node_id(("y", -1)))
    new_X = None
    X_node.add_data("valid_set", new_X)
    predictions = output.transform(ensemble, "valid_set")
