import numpy as np
import sklearn as skl
from Layer import Layer
class InputLayer(Layer):
    """description of class"""
    def __init__(self, n_features, N):
        self.A = np.zeros((n_features, N));
        return;

    def set_A(self, X):
        for i in range(X.shape[1]):
            self.A[i] = X.T[i];
        return;


