import numpy as np
from Layer import Layer

class OutputLayer(Layer):
    """description of class"""
    def __init__(self, N):
        self.Z = np.array([]*N);
        return;

    def set_Z(self, A_from_to_the_last_hidden_layer):
        self.Z = np.sign(A_from_to_the_last_hidden_layer);
        return;

    

