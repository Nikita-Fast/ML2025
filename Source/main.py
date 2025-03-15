import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils import check_array, check_scalar, check_random_state
from sklearn.utils.validation import check_is_fitted
import pandas as pd


class MyKMeans:
    def __init__(self, k: int):
        self._k = k
        self._max_iters = 50
        self._centroids = None

    def fit(self, X):
        self._centroids = self._get_initial_centroids(X)

        for i in range(self._max_iters):
            dist = np.sum(np.abs(X[:, None, :] - self._centroids) ** 2, axis=-1)
            clasters = np.argmin(dist, axis=-1)

            for i in range(self._k):
                c_new = np.mean(X[clasters == i], axis=0)
                delta = np.abs(c_new - self._centroids[i])
                self._centroids[i] = c_new

        return clasters

    def _get_initial_centroids(self, X):
        idx = np.random.randint(0, len(X), self._k)
        return X[idx]


class MyKMeans2(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=10, *, tol=1e-4, random_state=None):
        self.n_clusters = check_scalar(n_clusters, "n_clusters", int, min_val=1)
        self.tol = check_scalar(tol, "tol", float, min_val=0)
        self.random_state = check_random_state(random_state)

    def _get_labels(self, x):
        distances = np.sum(np.abs(x[:, None, :] - self.centroids) ** 2, axis=-1)
        labels = np.argmin(distances, axis=-1)
        return labels

    def predict(self, x):
        check_is_fitted(self)
        return self._get_labels(check_array(x))

    def fit(self, x, y=None):
        x = check_array(x)
        self.centroids = x[self.random_state.randint(0, len(x), self.n_clusters)]

        max_change = np.inf
        deltas = np.zeros(self.n_clusters, dtype=float)
        while max_change > self.tol:
            self.labels_ = self._get_labels(x)

            for i in range(self.n_clusters):
                c_new = np.mean(x[self.labels_ == i], axis=0)
                deltas[i] = np.linalg.norm(c_new - self.centroids[i])
                self.centroids[i] = c_new


            max_change = np.max(deltas)

        return self


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    df = pd.read_csv(r"C:\Users\user37\Downloads\fashion-mnist_train.csv")[:1000]
    df = df / 255
    X = df.drop('label', axis=1)
    Y = df['label']

    # X = np.random.rand(100, 2)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    k = 10
    kmeans = MyKMeans2(n_clusters=k)

    kmeans_copy = clone(kmeans)
    labels = kmeans_copy.fit_predict(X)

    # plt.figure(figsize=(8, 8))
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        edgecolor="none",
        alpha=0.7,
        s=40,
        cmap=matplotlib.colormaps['tab10'],
    )
    plt.colorbar()
    plt.title("t-SNE two-dimensional projection")
    plt.show()


    # plt.scatter(kmeans_copy.centroids[:, 0], kmeans_copy.centroids[:, 1], marker='*', s=250)
    # for i in range(k):
    #     plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'k={i}')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()
