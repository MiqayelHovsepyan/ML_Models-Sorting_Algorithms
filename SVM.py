import numpy as np

class SVM_Primal:
    def __init__(self, learning_rate=0.1, C=1.0, n_iters=1000):
        self.learning_rate = learning_rate
        self.C = C
        self.n_iters = n_iters

    def objective_function(self, X, y):
        hinge_losses = np.maximum(0, 1 - y * (X @ self.weights + self.bias))
        return (1/2)*np.sum(self.weights**2) + self.C * np.mean(hinge_losses)

    def fit(self, X, y):
        X_train = np.array(X)    
        y_train = np.array(y)    
        self.weights = np.random.randn(X_train.shape[1]) * 0.01
        self.bias = 0
        for _ in range(self.n_iters):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(X_train.shape[0]):
                X_i = X_train[i]
                y_i = y_train[i]

                margin_i = y_i * ((X_i @ self.weights) + self.bias)
                if margin_i >= 1:
                    dw = self.weights
                    db = 0
                else:
                    dw = self.weights - self.C * y_i * X_i
                    db = -self.C * y_i

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db       

    def predict(self, X_test):
        return np.sign(X_test @ self.weights + self.bias)            
    


class SVM_Dual:
    def __init__(self, C=1.0, kernel=None):
        self.C = C
        self.kernel = kernel

    def gram_matrix(self, X):
        X_train = np.array(X)
        n_samples = X_train.shape[0]
        K = np.zeros((n_samples,n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == None:
                    K[i, j] = np.dot(X_train[i], X_train[j])
                else:
                    K[i, j] = self.kernel(X_train[i], X_train[j])
        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.astype(float)

        # Compute Gram matrix
        K = self.gram_matrix(X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))

        # Inequality constraints: 0 <= alpha <= C
        G_std = -np.eye(n_samples)
        h_std = np.zeros(n_samples)
        G_slack = np.eye(n_samples)
        h_slack = np.ones(n_samples) * self.C

        G = matrix(np.vstack((G_std, G_slack)))
        h = matrix(np.hstack((h_std, h_slack)))

        # Equality constraint
        A = matrix(y.reshape(1, -1))
        b = matrix(0.0)

        # Solve QP
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).flatten()

        # Support vectors
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        # Compute bias
        self.bias = np.mean(
            [y_k - np.sum(self.alphas * self.sv_y * K[idx, sv])
            for idx, y_k in enumerate(y[sv])]
        )

        # Compute weight vector (linear case only)
        if self.kernel is None:
            self.weights = np.sum(self.alphas[:, None] * self.sv_y[:, None] * self.sv_X, axis=0)

    def predict(self, X):
        if self.kernel is None:
            return np.sign(X @ self.weights + self.bias)
        else:
            # Kernel case: sum over support vectors
            y_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha, y_sv, x_sv in zip(self.alphas, self.sv_y, self.sv_X):
                    s += alpha * y_sv * self.kernel(X[i], x_sv)
                y_pred[i] = s
            return np.sign(y_pred + self.bias)
            
