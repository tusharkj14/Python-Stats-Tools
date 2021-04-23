import numpy as np 

class NaiveBayes :

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._prior[idx] = X_c.shape[0] / float(n_samples)


    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp(- (x-mean)**2 / (2 * var))
        denom = np.sqrt(2* np.pi* var)
        return num/denom

    def predict_each(self, x):
        posteriors = []

        for idx, _ in enumerate(self._classes):
            prior = np.log(self._prior[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(prior + posterior)

        return self._classes[np.argmax(posteriors)]
    
    def predict(self, X):
        predicted_y = [self.predict_each(x) for x in X]
        return predicted_y

    