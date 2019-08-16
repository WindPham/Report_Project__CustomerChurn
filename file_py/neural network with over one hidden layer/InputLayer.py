import numpy as np
import sklearn as skl
from Layer import Layer
class InputLayer(Layer):
    """description of class"""
    def set_A(self, X):
        self.A = X.T;
        return;


