import numpy as np

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (-1/len(y_true)) * np.sum((y_true * np.log(y_pred)) + ((1-y_true) * np.log(1-y_pred)))
    

class Logistic_Regression:
    def __init__(self, learning_rate=0.1, threshold=0.5, n_iters=1000):
        self.learning_rate = learning_rate
        self.threshold = threshold    
        self.n_iters = n_iters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X_train = np.array(X)
        y_train = np.array(y).flatten()
        self.weights = np.random.randn(X_train.shape[1])*0.01
        self.bias = 0
        n_samples = X_train.shape[0]
        for _ in range(self.n_iters):
            z = X_train @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            dw = (1/n_samples) * X_train.T @ (y_pred - y_train)
            db = (1/n_samples) * np.sum(y_pred - y_train)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db    

    def predict_proba(self, X_test):
        z = X_test @ self.weights + self.bias 
        self.pred_prob = self.sigmoid(z)
        return self.pred_prob
    
    def predict(self, X_test):
        y_pred = np.array([1 if x > self.threshold else 0 for x in self.predict_proba(X_test)])
        return y_pred 