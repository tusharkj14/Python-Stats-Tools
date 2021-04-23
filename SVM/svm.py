import numpy as np 

class SVM:

    def __init__(self, learning_rate =0.001, lambda_param=0.01, n_iter=1000):
        self._lr = learning_rate
        self._lp = lambda_param
        self._ni = n_iter

        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <=0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self._ni):
            for idx, x in enumerate(X):
                cdn = y_[idx] * (np.dot(x, self.w) - self.b) >=1

                if cdn:
                    self.w -= self._lr * 2 * self._lp * self.w 
                
                else:
                    self.w -= self._lr * (2 * self._lp* self.w - np.dot(x, y_[idx]))
                    self.b -= self._lr * y_[idx]

    def predict(self, X):
        predicted_y = np.dot(X, self.w) - self.b 
        return np.sign(predicted_y)