import numpy as np;
from Layer import Layer

class HiddenLayer(Layer):
    """description of class"""
    def __init__(self, d):
        self.Z = np.zeros((d));
        self.A = np.zeros((d));
        return;
    def activation_function(self, s):
        return (s>0)*s;

    def Z_to_A(self):
        self.A = self.activation_function(self.Z);
        return;

    def set_Z(self, Z):
        self.Z=Z;
        return;

    

    