
import numpy as np
from InputLayer import InputLayer;
from HiddenLayer import HiddenLayer;
from OutputLayer import OutputLayer;
from Connection import Connection;

class Network(object):
    """description of class"""
    def __init__(self, learningrate, num_data_points, n_features , num_of_hidden_layers, array_num_of_unit_each_layer, Ws, bs):
        self.learning_rate = learningrate;
        self.n_class = Ws[-1].shape[1];
        self.N = num_data_points;
        self.features = n_features;
        self.L = num_of_hidden_layers + 1;
        self.input_layer = InputLayer(self.features, self.N);
        self.output_layer = OutputLayer(self.N, self.n_class);
        self.hidden_layers = np.array([]);
        self.connections = np.array([]);
        for i in range(num_of_hidden_layers):
            self.hidden_layers = np.append(self.hidden_layers, HiddenLayer(array_num_of_unit_each_layer[i]));
        for i in range(1, self.L+1):
            self.connections = np.append(self.connections, Connection());
            self.connections[i-1].set(Ws[i-1], bs[i-1], i-1, i);
        
        return;

    def print(self):
        for i in self.connections:
            i.print();
        return;

    def feed_forward(self, X):
        # set A of input_layer
        self.input_layer.set_A(X);
        # Z(1) = W(1).T * A(0) + b(1);
        self.hidden_layers[0].set_Z(self.connections[0].Z(self.input_layer.A) + self.connections[0].b);# start at 0
        # A(1) = activation_function (Z(1));
        self.hidden_layers[0].Z_to_A();
        # loop
        for i in range(1, len(self.hidden_layers)):
            # i = 1 ...... L-1; L = 3
            # Z(i) = W(i).T * A(i-1) + b(i);
            self.hidden_layers[i].set_Z(self.connections[i].Z(self.hidden_layers[i-1].A) + self.connections[i].b);
            # A(i) = activation_function (Z(i));
            self.hidden_layers[i].Z_to_A();
        # Z(L) = W(L) * A[L-1] + b(L);
        self.output_layer.set_Z(self.connections[self.L-1].Z(self.hidden_layers[self.L-2].A));
        # A(L) = activation_function(Z(L));
        self.output_layer.Z_to_A();
        return self.output_layer.A;
    
    def train(self, X, Y): 
        self.feed_forward(X);
        # calculate E(L) = 1/N * (Yhat - Y);
        E_L = 1/self.N * (self.output_layer.A - Y);
        pdJ_pdW_L = np.dot(self.hidden_layers[self.L-2].A, E_L.T); # pd(J)/pd(W_L);
        pdJ_pdb_L = np.sum(E_L, axis = 1, keepdims=True); # pd(J).pd(b_L);

        # Calculate E(L-1) = W(L-1) dot E(L).
        E_L_1 = np.dot(self.connections[self.L-1].W, E_L);
        pdJ_pdW_L_1 = np.dot(self.hidden_layers[self.L - 3].A, E_L_1.T);# pd(J)/pd(W_L_1);
        pdJ_pdb_L_1 = np.sum(E_L_1, axis = 1, keepdims=True);# pd(J)/pd(b_L_1);
        
        # Calculate E(L-2) = W(L-2) dot E(L-1).
        E_L_2 = np.dot(self.connections[self.L-2].W, E_L_1);
        pdJ_pdW_L_2 = np.dot(self.input_layer.A, E_L_2.T);# pd(J)/pd(W_L_2);
        pdJ_pdb_L_2 = np.sum(E_L_2, axis = 1, keepdims=True);# pd(J)/pd(b_L_2);
        #print(pdJ_pdW_L);
        #print(pdJ_pdb_L);
        #print(pdJ_pdW_L_1);
        #print(pdJ_pdb_L_1);
        #print(pdJ_pdW_L_2);
        #print(pdJ_pdb_L_2);
        #
        # w += -eta * pd(j)/pd(w)
        # b += -eta + pd(j)/pd(w)
        self.connections[self.L - 1].update(self.learning_rate, pdJ_pdW_L  , pdJ_pdb_L  );
        self.connections[self.L - 2].update(self.learning_rate, pdJ_pdW_L_1, pdJ_pdb_L_1);
        self.connections[self.L - 3].update(self.learning_rate, pdJ_pdW_L_2, pdJ_pdb_L_2);

        return;

    def predict(self, X , Y):
        pred = self.feed_forward(X);
        Y_new = 1.0*np.argmax(pred, axis = 0);

        return np.sum(Y_new == Y);

        return;

    


