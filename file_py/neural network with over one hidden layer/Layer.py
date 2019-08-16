import numpy as np
import sklearn as skl
from abc import ABC
#from abc import abstractmethod

class Layer(ABC):
    """description of class"""
    def __init__(self):
        self.Z=np.array([]);
        self.A=np.array([]);
        return;

    def set_Z(self, Z):
        self.Z=Z;
        return;

    def Z_to_A(self):
        self.A = self.activation_function(self.Z);
        return;
    
    def activation_function(self, s):
        pass;