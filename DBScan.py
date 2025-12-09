import numpy as np

class DBScan:
    def __init__(self, eps=0.5, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)

        cluster_id = 0
        visited = set()
        for i in range(n_samples):
            if i in visited:
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                visited.add(i)
                continue

            visited.add(i)
            self.labels[i] = cluster_id

            self._expand_cluster(X, neighbors, visited, cluster_id)

            cluster_id += 1
        
        return self.labels
    
    def _get_neighbors(self, X, i):
        distances = np.linalg.norm(X - X[i], axis=1)
        return np.where(distances < self.eps)[0]

    def _expand_cluster(self, X, neighbors, visited, cluster_id):
        queue = list(neighbors)

        while queue:
            q_index = queue.pop(0)

            if q_index in visited:
                if self.labels[q_index] == -1:
                    self.labels[q_index] = cluster_id
                continue

            visited.add(q_index)
            self.labels[q_index] = cluster_id

            q_neighbors = self._get_neighbors(X, q_index)

            if len(q_neighbors) >= self.min_samples:
                for r_index in q_neighbors:
                    if r_index not in visited:
                        queue.append(r_index)
                    elif self.labels[r_index] == -1:
                        self.labels[r_index] = cluster_id