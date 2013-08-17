import numpy as np

from sklearn.linear_model import LogisticRegression

import SETTINGS
from transform import SklearnBridge
from feature_store import FeatureDependency, FeatureStore

if 0:
    # adding data
    SETTINGS.FEATURE_STORE.CV_FOLDS = 2
    trn = SklearnBridge(clf=LogisticRegression())
    y = np.random.randint(0, 2, (10, 1))

    import os
    os.system('rm feature.db')
    os.system('rm -rf __TRAIN__ test test2')

    store = FeatureStore()
    input1_0 = store.input_node(np.ones((10, 10)), "input1")
    input1_1 = store.input_node(np.random.randn(10, 10), "input1")
    input2 = store.input_node(y, "input2", tags=('doo', 'foo'))
    X_dep = FeatureDependency(parent_labels=[("input1", 0)])
    y_dep = FeatureDependency(parent_tags=["doo"])
    node = store.create("logRes", trn, ["prediction"], X_dep, y_dep, stratified=False)
    cv = store.fit_transform(node)

    input1_0.add_data('test', np.ones((10, 10)))
    input1_0.add_data('test2', np.random.randn(10, 10))
    print store.transform(node, data_name="test")
    print store.transform(node, data_name="test2")
    print store.transform(node)


if 0:
    # manually querying data
    store = FeatureStore()
    node = store.node(4)
    print store.fit_transform(node)[0]
    print store.transform(node, data_name="test")
    print store.transform(node, data_name="test2")
    print store.transform(node)
    print store.transform(node)

if 1:
    # querying data
    store = FeatureStore()
    print store.data(("input1", 0))
    print store.data(("input1", 1))
    print store.data(("input2", 0))
    print store.data(("logRes", 0))
