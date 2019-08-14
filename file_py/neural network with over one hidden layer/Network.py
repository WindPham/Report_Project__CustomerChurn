
import numpy as np
from InputLayer import InputLayer;
from HiddenLayer import HiddenLayer;
from OutputLayer import OutputLayer;
from Connection import Connection;

class Network(object):
    """description of class"""
    def __init__(self, num_data_points, n_features , num_of_hidden_layers, array_num_of_unit_each_layer, Ws, bs):
        self.N = num_data_points;
        self.features = n_features;
        self.L = num_of_hidden_layers + 1;
        self.input_layer = InputLayer(self.features, self.N);
        self.output_layer = OutputLayer(self.N);
        self.hidden_layers = np.array([]);
        self.connections = np.array([]);
        for i in range(num_of_hidden_layers):
            self.hidden_layers = np.append(self.hidden_layers, HiddenLayer(array_num_of_unit_each_layer[i]));
        for i in range(1, num_of_hidden_layers + 1):
            self.connections = np.append(self.connections, Connection());
            self.connections[i-1].set(Ws[i-1], bs[i-1], i-1, i);
        return;

    def print(self):
        for i in self.connections:
            i.print();
        return;

    def feed_forward(self, X):
        self.input_layer.set_A(X);
        self.hidden_layers[0].set_Z(self.connections[0].Z(self.input_layer.A) + self.connections[0].b);
        self.hidden_layers[0].Z_to_A()
        for i in range(1, len(self.hidden_layers)):
            self.hidden_layers[i].set_Z(self.connections[i].Z(self.hidden_layers[i-1].A) + self.connections[i-1].b);
            self.hidden_layers[i].Z_to_A();
        self.output_layer.set_Z(self.hidden_layers[self.L-2].A);
        return self.output_layer.Z;
    
    def train(self):
        return;

    


