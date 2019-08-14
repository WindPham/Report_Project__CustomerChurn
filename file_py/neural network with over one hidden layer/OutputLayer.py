import numpy as np
from Layer import Layer

class OutputLayer(Layer):
    """description of class"""
    def __init__(self, N, n_class):
        self.Z = np.array([]*N);
        self.A = np.array([]*N);
        return;

    def activation_function(self, s):
        exps = np.exp(s);
        return exps / np.sum(exps, axis = 0);

    def set_Z(self, Z):
        self.Z=Z;
        return;

    def Z_to_A(self):
        self.A = self.activation_function(self.Z);
        return;

    

