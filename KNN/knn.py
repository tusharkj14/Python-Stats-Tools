import numpy as np 
from collections import Counter

def eculidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNearestNeighbors:

    def __init__(self, k_=5):
        self.k_ = k_

    def fit(self, X, y):

        if isinstance(X, (np.float64, float)):
            dataset = X
        else:
            dataset = X.to_numpy('float64')

        self.X_train = dataset
        self.y_train = y 

    def predict_each(self, x):
        distances = [eculidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k_]

        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        
        return Counter(k_nearest_labels).most_common(1)[0][0]

    def predict(self, X):

        if isinstance(X, (np.float64, float)):
            dataset = X
        else:
            dataset = X.to_numpy('float64')

        predicted_y = [self.predict_each(x) for x in dataset]
        return np.array(predicted_y)