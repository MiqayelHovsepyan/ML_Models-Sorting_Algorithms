import numpy as np

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

class Linear_Regression:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y).flatten()
        self.weights = np.random.rand(self.X_train.shape[1])
        self.bias = 0
        self.n_samples = self.X_train.shape[0]
        for _ in range(self.n_iters):
            y_pred = (self.X_train @ self.weights + self.bias).flatten()
            dw = (2/self.n_samples) * (self.X_train.T @ (y_pred - self.y_train))
            db = (2/self.n_samples) * np.sum(y_pred - self.y_train)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = (X_test @ self.weights + self.bias).flatten()        
        return y_pred
    

    class Linear_Regression_Ridge:
        def __init__(self, learning_rate=0.01, alpha=0.1, n_iters=1000):
            self.learning_rate = learning_rate
            self.n_iters = n_iters
            self.alpha = alpha

        def fit(self, X, y):
            X_train = np.array(X)    
            y_train = np.array(y).flatten()    
            self.weights = np.random.randn(X_train.shape[1])*0.01
            self.bias = 0
            n_samples = X_train.shape[0]
            for _ in range(self.n_iters):
                y_pred = (X_train @ self.weights + self.bias).flatten()
                dw = (2/n_samples) * (X_train.T @ (y_pred - y_train)) + (2*self.alpha * self.weights)
                db = (2/n_samples) * np.sum(y_pred - y_train)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        def predict(self, X_test):
            X_test = np.array(X_test)
            y_pred = (X_test @ self.weights + self.bias).flatten()
            return y_pred
        

    class Linear_Regression_Lasso:
        def __init__(self, learning_rate=0.1, betta=0.01, n_iters=1000):
            self.learning_rate = learning_rate
            self.betta = betta
            self.n_iters = n_iters

        def fit(self, X, y):
            X_train = np.array(X)
            y_train = np.array(y).flatten()
            self.weights = np.random.randn(X_train.shape[1]) * 0.01
            self.bias = 0
            n_samples = X_train.shape[0]
            for _ in range(self.n_iters):
                y_pred = (X_train @ self.weights + self.bias).flatten()
                dw = (2/n_samples) * (X_train.T @ (y_pred - y_train)) + self.betta*np.sign(self.weights)
                db = (2/n_samples) * np.sum(y_pred - y_train)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        def predict(self, X_test):
                X_test = np.array(X_test)
                y_pred = (X_test @ self.weights + self.bias).flatten()
                return y_pred