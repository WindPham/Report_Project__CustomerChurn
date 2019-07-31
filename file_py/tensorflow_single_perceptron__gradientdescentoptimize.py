
import tensorflow as tf
import numpy as np
import sklearn as skl
import scipy as sci
import pandas as pd
import seaborn as sb
import matplotlib as mplt
import matplotlib.pyplot as plt
import time
np.random.seed(int(time.time())%100);
def fun(x):
    return 2*x[:,0] - 3*x[:,1] + 1;
def generate_data(N):
    X_train = np.random.uniform(-10.0, 10.0 , size = (N, 2));
    Y_train = np.sign(fun(X_train));
    rd_noise = np.random.choice(100,10000)
    Y_train[rd_noise] = -1*Y_train[rd_noise];
    plt.scatter(X_train[:,0], X_train[:,1], c = Y_train);
    
    plt.show()
    return X_train, Y_train;

N = 10000
X_train, Y_train = generate_data(N)

W = tf.Variable(tf.zeros([2,1]));
b = tf.Variable([-1], dtype=tf.float32);

x = tf.compat.v1.placeholder(tf.float32, [N, 2]);
y = tf.compat.v1.placeholder(tf.float32);

model = tf.matmul(x,W) + b;

#cho nay khong biet cho loss = ?
loss_value = tf.reduce_sum(tf.abs(tf.sign(model)-y)/2);

gra_op = tf.compat.v1.train.GradientDescentOptimizer(0.01);

train = gra_op.minimize(loss_value);

init = tf.compat.v1.global_variables_initializer();

sess = tf.compat.v1.Session();
sess.run(init);

for i in range(1000):
    sess.run(train, {x: X_train, y: Y_train});

cW, cb, c_loss = sess.run([W, b, loss_value], {x: X_train, y: Y_train});

print("W: %s %s, b: %s, loss: %s"%(cW[0], cW[1], cb, c_loss));
