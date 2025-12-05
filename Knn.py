import numpy as np
import pandas as pd
def euclidean(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2))

class KNN_Regressor:
    def __init__(self, neighbors, distance_metric=euclidean):
        self.neighbors = neighbors
        self.distance_metric = distance_metric
    def fit(self, X,y):
        self.X_train = np.array(X) 
        self.y_train = np.array(y)   

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])
    
    def _predict(self, x):
        distances = [self.distance_metric(x, x_train) for x_train in self.X_train]
        idx = np.argsort(distances)[:self.neighbors]
        prediction = [self.y_train[i] for i in idx] 
        prediction = sum(prediction)/self.neighbors
        return prediction
    
class KNN_Classification:
    def __init__(self, neighbors, distance_metric=euclidean):
        self.neighbors = neighbors
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = np.array(X)  
        self.y_train = np.array(y)  

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        distances = [self.distance_metric(x,x_train) for x_train in self.X_train]
        idx = np.argsort(distances)[:self.neighbors]
        prediction = pd.Series(self.y_train[idx]).mode()[0]
        return prediction