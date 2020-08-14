import numpy as np

class Pca:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X = (X - self.mean)
        cov = np.cov(X.T)
        eigenval, eigenvect = np.linalg.eig(cov)
        eigenvect = eigenvect.T

        # indices of the eigenvalues in decreasing order
        idxs = np.argsort(eigenval)[::-1]
        eigenvalues = eigenval[idxs]
        eigenvectors = eigenvect[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        X = (X - self.mean)
        return np.dot(X,self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



