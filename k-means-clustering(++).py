import numpy as np

def euclidean(x1, x2):
    return np.linalg.norm(np.array(x1)-np.array(x2))

class Kmeans:
    def __init__(self, k=3, max_iters=300, init='kmeans++', n_init=10, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_samples = self.X.shape[0]
    def _init_centroids(self):
        if self.init == 'kmeans++':
            random_idx = np.random.choice(self.n_samples, 1)[0]
            centroids = [self.X[random_idx]]
            distances = np.full(self.n_samples, np.inf)
            for _ in range(1, self.k):
                for i in range(self.n_samples):
                    dist = np.sum((self.X[i] - centroids[-1])**2)
                    if dist < distances[i]:
                        distances[i] = dist
                
                prob = distances / distances.sum()
                next_idx = np.random.choice(self.n_samples, p=prob)
                centroids.append(self.X[next_idx])
        else: # init == 'random'
            random_idxs = np.random.choice(self.n_samples, self.k, replace=False)
            centroids = self.X[random_idxs]

        return np.array(centroids)
    
    def _compute_distances(self):
        self.distances = np.zeros((self.n_samples, self.k))
        for i in range(self.n_samples):
            for j in range(self.k):
                self.distances[i, j] = euclidean(self.X[i], self.centroids[j])
        return self.distances
    
    def _assign_clusters(self):
        self.labels = np.argmin(self.distances, axis=1)
        return self.labels
    
    # def _
    def fit(self, X):
        self.X = np.array(X)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        for _ in range(self.n_init):
            self.centroids = self._init_centroids()
            for i in range(self.max_iters):
                pass




#Basic
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n_samples = X.shape[0]

        # Step 1: initialize centroids randomly from samples
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            # Step 2: assign points to nearest centroid
            clusters = self._assign_clusters(X)

            # Step 3: compute new centroids
            new_centroids = np.array([X[clusters == i].mean(axis=0) 
                                      for i in range(self.k)])

            # Check for convergence (if centroids don't change)
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.zeros((self.n_samples, self.k))
        for i in range(self.n_samples):
            for j in range(self.k):
                distances[i, j] = euclidean(self.X[i], self.centroids[j])
        return distances
        # distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        # return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)


