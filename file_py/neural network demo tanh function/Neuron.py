import numpy as np
class Neuron(object): # each neuron is a cell or a cell for a layer
    """

    attribute:  bias: float
                weights: np.array();
                inputs: np.array();
                output: float;
    method:
                tanh(s: float): float --> active function.
                wT_dot_x( inputs: np.array() ): float --> include: init self.inputs, self.output.

    """

    def __init__(self, bias):
        self.bias = bias;
        self.weights = np.array([]);
        return ;
    def bias(self):
        return self.bias;

    # tanh function();
    def tanh(self, s): # s = wT.x;
        # (1-e^-s)/(1+e^-s);
        return ( np.exp(s) - np.exp(-s) ) / ( np.exp(s) + np.exp(-s) );

    # calculate wT.x
    def wT_dot_x(self, inputs):
        # wT.x = w1.x1 + w2.x2 + w3.x3 + ...... + wd.xd;
        self.inputs = inputs; # set attribute inputs = inputs
        self.output = np.sum(self.inputs * self.weights);
        return self.output + self.bias; # wT.x + b

    # calculate pd(J)/pd(z_j):
    def pd_J__pd_zj(self, target_output):
        # pd(J)/pd(z_j) = pd(J)/pd(yj) * pd(yj)/pd(zj);
        return self.pd_J__pd_yj(target_output)*self. d_yj__d_zj()

    # calculate pd(J)/pd(y_j):
    def pd_J__pd_yj(self, target_output):

        return -(target_output - self.output); # -(target - y)

    # calcuclate pd(y_j)/pd(z_j);
    def d_yj__d_zj(self):

        return 1 - (self.tanh(self.output))**2; # 1 - tanh^2

    def mean_square_error(self, target_output):

        return 0.5*(target_output - self.output)**2;

    # calculate pd(z_j)/pd(w_i):
    def pd_zj__pd_wi(self, index): return self.inputs[index];