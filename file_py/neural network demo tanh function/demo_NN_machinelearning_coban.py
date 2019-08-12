import numpy as np
import math as math
import scipy as scp
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as skl
import Neuron as unit
import NeuralNetwork as net
import NeuronLayer as layer



if __name__=='__main__':

    neural_net = net.NeuralNetwork( 2, 
                                    2, 
                                    2, 
                                    hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], 
                                    hidden_layer_bias=0.35, 
                                    output_layer_weights=[0.4, 0.45, 0.5, 0.55], 
                                    output_layer_bias=0.6);
    for i in range(100):
        neural_net.train([0.05, 0.1], [0.01, 0.99]);
        print("%d -- %0.20f"%(i, neural_net.total_error([[[0.05, 0.1], [0.01, 0.99]]])))
        