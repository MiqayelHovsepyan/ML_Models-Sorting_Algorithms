import numpy as np
from numpy.linalg import eig

class SVD:
    def __init__(self, A, full_matrices=True):
        A = np.array(A)
        if A.ndim != 2:
            raise ValueError("Only 2D arrays supported")

        self.A = A
        self.n, self.m = A.shape
        self.full_matrices = full_matrices

        AtA = A.T @ A
        vals, vecs = eig(AtA)

        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        self.singular_values = np.sqrt(np.abs(vals))
        self.Vt = vecs.T
        self.V = vecs

        self.U = np.zeros((self.n, len(self.singular_values)))
        for i in range(len(self.singular_values)):
            if self.singular_values[i] > 1e-10:
                self.U[:, i] = (A @ self.V[:, i]) / self.singular_values[i]

        if full_matrices:
            self.S = np.zeros((self.n, self.m))
            for i in range(len(self.singular_values)):
                self.S[i, i] = self.singular_values[i]
        else:
            self.S = np.diag(self.singular_values)

    def get_U(self):
        return self.U

    def get_Vt(self):
        return self.Vt

    def get_S(self):
        return self.S
