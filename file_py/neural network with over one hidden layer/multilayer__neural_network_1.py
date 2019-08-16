

import numpy as np
from Network import Network as net

def read_data(filename):
    file = open(filename, "r")
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


def f(x):
    return x[:,0] - x[:, 1];

def generate_data(N):
    X = np.random.rand(N,2);
    Y = np.sign(f(X));
    return X, Y;

def create_w (r, c):

    return np.random.uniform(low=-0.06, high=0.06, size=(r,c))

def create_bs(n_b):

    return np.random.uniform(low=0.001, high=0.3, size = (n_b,));

def create_ws(n_layers, ws):
    res = [];
    for i in range(0,n_layers):
        res.append(create_w(ws[i][0], ws[i][1]));
    return res;

if __name__=='__main__':
    
    #print(create_w(2,5));

    W1 = np.array([[0.002, 0.025, 0.03, -0.0275, -0.07], [-0.021, -0.024, -0.028, 0.018, 0.001]]);
    b1 = 0.0015
    
    W2 = np.array([[0.05, 0.02, -0.01], [0.03, 0.025, 0.04], [0.015, -0.035, -0.06], [-0.02, -0.027, 0.032], [-0.018, 0.0026, -0.05] ]);
    b2 = 0.23
    
    W3 = np.array([[0.023, 0.016], [-0.042 , -0.01], [0.026, -0.075]]);
    b3 = 0.14
    
    Ws = np.array([W1, W2, W3]);
    bs = np.array([b1, b2, b3]);
    #ws = [(2, 11), (11, 10), (10, 9), (9, 8), (8, 7), (7, 6), (6, 5), (5, 4), (4, 3), (3, 2)];
    #Ws = np.array(create_ws(10, ws));
    #bs = np.array(create_bs(10));

    X, Y = read_data('input.txt');
    Y[Y==-1.0]=0;
    neural_network = net(0.15325, X.shape[1], 2, Ws, bs);

    neural_network.train_loop(X, Y, 1000, mode = 'b', mini_batch = 10);

    print(str(neural_network.predict(X, Y)*100.0/500.0) + "% on train");


    X_new, Y_new = read_data("val_input.txt")
    Y_new[Y_new==-1.0]=0;
    print(str(neural_network.predict(X_new[0:100], Y_new[0:100])*100.0/100.0) + "% on val");


    x, y = generate_data(500000);
    y[y==-1]=0;
    print(str(neural_network.predict(x,y)*100.0/500000.0) + "% on random set");