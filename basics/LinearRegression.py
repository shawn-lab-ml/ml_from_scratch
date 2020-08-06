import numpy as np

class LinearRegression:
    def __init__(self, lr = 1e-3, max_iter = 1000):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.display_MSE = True

    def fit(self, X, y):
        # Implementation of a gradient descent as we must minimize the MSE
        n_obs, n_feat = X.shape

        self.w = np.random.rand(n_feat) - 0.5
        self.b = 0

        for _ in range(self.max_iter):
            # the formula for gradient descent is w = w - a*dw where dw is
            # the derivative of the cost function (MSE) with respect to
            # the weights

            y_pred = np.dot(X, self.w) + self.b
            self.w -= self.lr * np.dot(X.T,(y_pred -y))/ n_obs
            self.b -= self.lr * (y_pred - y) / n_obs

    def predict(self, X):
        return np.dot(X,self.w) + self.b

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)
