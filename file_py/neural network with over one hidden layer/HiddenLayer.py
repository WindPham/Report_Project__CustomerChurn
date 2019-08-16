import numpy as np;
from Layer import Layer

class HiddenLayer(Layer):
    def activation_function(self, s):
        return (s>0)*s;


    

    