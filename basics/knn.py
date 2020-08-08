import numpy as np


class Knn:
    def __init__(self, k=5, measure = "l2"):
        assert (measure == "l2" or measure == "l1"), "Choose correct measure"
        self.k = k
        self.X = None
        self.y = None
        self.measure = measure
        self.measures = {"l2": euclidean_distance,
                         "l1": manhattan_distance}

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return [self.predict_labels(obs) for obs in X]

    def predict_labels(self, obs):
        self


class KnnRegressor(Knn):
    def __init__(self, k=5,  measure="l2"):
        super().__init__(k, measure)

    def predict_labels(self, obs):
        measure = self.measures[self.measure]
        distances = [measure(obs, train_obs) for train_obs in self.X]
        # np.argsort is O(n*log(n)) time complexity
        kni = np.argsort(distances, )[:self.k]
        knn = [self.y[idx] for idx in kni]
        return np.mean(knn)


class KnnClassifier(Knn):
    def __init__(self, k=5, measure="l2"):
        super().__init__(k, measure)

    def predict_labels(self, obs):
        measure = self.measures[self.measure]
        distances = [measure(obs, train_obs) for train_obs in self.X]
        # np.argsort is O(n*log(n)) time complexity
        kni = np.argsort(distances, )[:self.k]
        knn = [self.y[idx] for idx in kni]

        return self.most_common(knn)


    def most_common(self,arr):
        h = dict()

        for e in arr:
            if e in h.keys():
                h[e] += 1
            else:
                h[e] = 1

        max_count = h[arr[0]]
        mc_elem = arr[0]

        for e in arr[1:]:
            if h[e] > max_count:
                mc_elem = e
                max_count = h[e]
        return mc_elem


def euclidean_distance(obs1, obs2):
    dist = 0.0
    for feat in range(len(obs1)):
        dist += (obs1[feat] - obs2[feat]) ** 2
    return np.sqrt(dist)

def manhattan_distance(obs1, obs2):
    dist = 0.0
    for feat in range(len(obs1)):
        dist += np.abs(obs1[feat] - obs2[feat])
    return dist