from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import ElasticNet

clf1 = MiniBatchKMeans(1000)
clf2 = ElasticNet()


class TransformPredictMachine(object):

    def __init__(self, transformer, predicter):
        self.transformer = transformer
        self.predicter = predicter

    def fit(self, X, y):
        tmp = self.transformer.fit_transform(X, y)
        self.predicter.fit(tmp, y)

    def predict(self, X):
        tmp = self.transformer.transform(X)
        return self.predicter.predict(tmp)


def kmeans_linear_model(n_clusters=2, alpha=1.0, l1_ratio=0.5, fit_intercept=True, positive=False):
    transformer = MiniBatchKMeans(n_clusters=n_clusters, compute_labels=False, random_state=None)
    predicter = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, positive=positive)
    return TransformPredictMachine(transformer, predicter)
