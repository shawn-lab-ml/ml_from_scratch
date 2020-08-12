import numpy as np

class Logreg:
    def __init__(self, lr=1e-3, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_obs, n_feat = X.shape
        self.w = np.zeros(n_feat)
        self.b = 0

        for _ in range(self.max_iter):
            linear_model = np.dot(X,self.w) + self.b
            y_pred = self.sigmoid(linear_model)

            dw = np.sum(np.dot(X.T, (y_pred-y)))/n_obs
            db = np.sum((y_pred-y))/n_obs

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X, proba = False):
        linear_model = np.dot(X,self.w) + self.b
        y_pred = self.sigmoid(linear_model)

        if proba == True:
            return y_pred

        y_pred_c = [1 if e > 0.5 else 0 for e in y_pred]
        return y_pred_c

    def sigmoid(self, x):
        return 1/ (1+np.exp(-x))


