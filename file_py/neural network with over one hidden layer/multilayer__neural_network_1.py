

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


if __name__=='__main__':
    
    W1 = np.array([[0.002, 0.025, 0.03, -0.0275], [-0.021, -0.024, -0.028, 0.018]]);
    b1 = 0.0015
    
    W2 = np.array([[0.05, 0.02, -0.01], [0.03, 0.025, 0.04], [0.015, -0.035, -0.06], [-0.02, -0.027, 0.032]]);
    b2 = 0.23
    
    W3 = np.array([[0.023, 0.016], [-0.042 , -0.01], [0.026, -0.075]]);
    b3 = 0.14
    
    Ws = np.array([W1, W2, W3]);
    bs = np.array([b1, b2, b3]);


    X, Y = read_data('input.txt');
    Y[Y==-1.0]=0;
    neural_network = net(0.0673, X.shape[0], X.shape[1], 2, [4,3], Ws, bs);
        #net. learning_rate = 0.0673;
        #net. n_class = 2;
        #net. N = N;
        #net. features = 2
        #net. L = 3;
    #neural_network.print();
    #print(neural_network.feed_forward(X).T);
    for i in range(1000):
        #print(i);
        neural_network.train(X, Y);

    print(str(neural_network.predict(X, Y)*100.0/500.0) + "% on train");


    X_new, Y_new = read_data("val_input.txt")
    Y_new[Y_new==-1.0]=0;
    print(str(neural_network.predict(X_new, Y_new)*100.0/500.0) + "% on val");


    x, y = generate_data(500);
    y[y==-1]=0;
    print(str(neural_network.predict(x,y)*100.0/500.0) + "% on random set");