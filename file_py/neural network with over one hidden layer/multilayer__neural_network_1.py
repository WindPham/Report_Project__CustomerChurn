

import numpy as np
from Network import Network as net

def read_data(file = open("input.txt", "r")):
    data = file.read();
    data = data.strip('\n').split('\n');
    X = np.zeros((len(data), 2));
    Y = np.zeros((len(data)));
    for i in range(len(data)):
        j = data[i].strip('\n').split();
        for k in range(len(j)-1):
            X[i][k] = float(j[k]);
        Y[i] = float(j[2]);
    return X, Y;



if __name__=='__main__':
    
    W1 = np.array([[0.2, 0.25, 0.3, -0.275], [-0.21, -0.24, -0.28, 0.18]]);
    b1 = 0.1

    W2 = np.array([[0.5, 0.2, -0.1], [0.3, 0.25, 0.4], [0.15, -0.35, -0.6], [-0.2, -0.27, 0.32]]);
    b2 = 0.2

    W3 = np.array([0.23, 0.16, -0.42]);
    b3 = 0.14

    Ws = np.array([W1, W2, W3]);
    bs = np.array([b1, b2, b3]);

    X, Y = read_data();

    neural_network = net(X.shape[0], X.shape[1], 2, [4,3], Ws, bs);
    #neural_network.print();
    print(neural_network.feed_forward(X[:,0:2]));

