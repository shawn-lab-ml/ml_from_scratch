import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = None
        self.likelihoods = None
        self.classes = None

    def fit(self, X, y):
        n_obs, n_feat = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.priors = np.zeros(n_classes)
        self.likelihoods = np.zeros((n_classes, n_feat))


        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = X.shape[0] / n_obs
            self.likelihoods[idx,:] = (X_c.sum(axis=0) + self.alpha) / (X.sum(axis=0) + self.alpha)

    def predict(self, X):
        return [self.predict_observation(x) for x in X]

    def predict_observation(self, X):
        posteriors = []
        # using log as it easier to calculate (transforms product into sums)
        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.priors[idx])
            log_likelihoods = np.log(self.likelihoods[idx,:]) * X
            log_posterior = log_prior + log_likelihoods
            posteriors.append(log_posterior)
        return self.classes[np.argmax(posteriors)]





