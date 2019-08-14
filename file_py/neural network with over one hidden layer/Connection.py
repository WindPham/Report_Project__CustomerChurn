import numpy as np

class Connection(object):
    """description of class"""
    def __init__(self):
        self.W = np.zeros([]);
        self.b = 1.0;
        self.trade = (0,0);
        return;

    def set(self, W, b, l1, l2):
        self.W=W;
        self.b=b;
        self.trade = (l1, l2);
        return;

    def Z(self, A_of_first_layer):
        print(self.W.T.shape)
        return np.dot(self.W.T, A_of_first_layer);

    def print(self):
        print("W:\n %s"%(self.W));
        print("b: %s"%(self.b));
        print("trade: %s"%str(self.trade));
    
        return;


