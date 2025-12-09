import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)

        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        self.components_ = Vt[:self.n_components, :]

        self.explained_variance_ = (S[: self.n_components]**2) / (n_samples-1)

        return self
    
    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_ + self.mean_
    



X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0]])

pca = PCA(n_components=1)
pca.fit(X)
X_reduced = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

print("Components:\n", pca.components_)
print("Explained variance:\n", pca.explained_variance_)
print("Reduced X:\n", X_reduced)
print("Reconstructed X:\n", X_reconstructed)