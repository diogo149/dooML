"""
todo:
    if x is too large, take a sample and compute gap statistic on that
"""

import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import RandomizedPCA, PCA

import SETTINGS


def gap_statistic(x, random_datasets=64):
    """
    Returns the gap statistic of the data set. Keeps increasing the number of clusters until the maximum gap statistic is more than double the current gap statistic.
    http://blog.echen.me/2011/03/19/counting-clusters/
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 2

    if x.shape > SETTINGS.GAP_STATISTIC.RANDOMIZED_PCA_THRESHOLD:
        pca = RandomizedPCA(SETTINGS.GAP_STATISTIC.RANDOMIZED_PCA_THRESHOLD)
    else:
        pca = PCA()

    pca.fit(x)
    transformed = pca.transform(x)

    reference_datasets = [pca.inverse_transform(generate_random_dataset(transformed)) for _ in range(random_datasets)]

    max_gap_statistic = -1
    best_num_clusters = 1

    for num_clusters in range(1, x.shape[0] + 1):
        kmeans = MiniBatchKMeans(num_clusters)
        kmeans.fit(x)

        trained_dispersion = dispersion(kmeans, x)

        random_dispersions = [dispersion(kmeans, data) for data in reference_datasets]

        gap_statistic = np.log(sum(random_dispersions) / random_datasets) - np.log(trained_dispersion)

        if gap_statistic > max_gap_statistic:
            max_gap_statistic = gap_statistic
            best_num_clusters = num_clusters

        if gap_statistic < max_gap_statistic * SETTINGS.GAP_STATISTIC.MAXIMUM_DECLINE:
            break
        if num_clusters > best_num_clusters + SETTINGS.GAP_STATISTIC.NUM_CLUSTERS_WITHOUT_IMPROVEMENT:
            break

    return best_num_clusters


def dispersion(kmeans, data):
    return np.sum(kmeans.transform(data))


def generate_random_dataset(data):
    """
    generates a random dataset uniformly in the bounding box of the input data
    """
    random_data = np.random.uniform(size=data.shape)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    new_data = np.subtract(np.multiply(random_data, maxs - mins), mins)
    return new_data
