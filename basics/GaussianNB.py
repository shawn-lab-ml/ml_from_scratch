import numpy as np

class GaussianNB:

    def __init__(self):
        self.priors = None
        self.classes = None
        self.var = None
        self.mean = None

    def fit(self, X, y):
        n_obs, n_feat = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_feat), dtype = np.float32)
        self.var = np.zeros((n_classes, n_feat), dtype=np.float32)
        self.priors = np.zeros(n_classes, dtype=np.float32)
        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.priors[idx] = len(X_c)/ n_obs
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = np.square(np.sum(X - X_c.mean(axis=0), axis=0))/n_obs

    def predict(self, X):
        return [self.predict_obs(x) for x in X]

    def predict_obs(self, x_obs):
        posteriors = []
        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.priors[idx])
            log_likelihood = self.gaussian_pdf(x_obs, idx)
            log_likelihood = np.sum(np.log(log_likelihood))
            log_posterior = log_likelihood + log_prior
            posteriors.append(log_posterior)
        return self.classes[np.argmax(posteriors)]

    def gaussian_pdf(self, x_obs, idx):
        mean_c = self.mean[idx, :]
        var_c = self.var[idx, :]
        denominator = np.sqrt(var_c*np.pi*2)
        numerator = np.exp(-(x_obs-mean_c)**2/(2*var_c))
        return numerator/denominator