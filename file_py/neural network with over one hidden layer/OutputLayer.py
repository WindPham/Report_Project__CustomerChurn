import numpy as np
from Layer import Layer

class OutputLayer(Layer):
    def activation_function(self, s):
        exps = np.exp(s);
        return exps / np.sum(exps, axis = 0);
    

