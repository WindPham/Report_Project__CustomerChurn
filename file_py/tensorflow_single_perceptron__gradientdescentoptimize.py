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
# this is target function
def fun(x):
    return 2*x[:,0] - 3*x[:,1] + 1;#y = 2/3 x + 1/3
# generate a dataset with N points.
def generate_data(N):
    X_train = np.random.uniform(-10.0, 10.0 , size = (N, 2));
    Y_train = np.sign(fun(X_train));
    return X_train, Y_train;
# init  N = 100, training set is included X_train and Y_train
N = 100
X_train, Y_train = generate_data(N)
# init W and b
W = tf.Variable(tf.zeros([2,1])); #[0
                                 #  0]

b = tf.Variable([-1], dtype=tf.float32);# bias

x = tf.compat.v1.placeholder(tf.float32, [N, 2]);
y = tf.compat.v1.placeholder(tf.float32);

model = tf.matmul(x,W) + b;

# init loss_value? Right or Wrong?
loss_value = tf.reduce_mean(tf.multiply(-y, model));
# set learning rate = 1
gra_op = tf.compat.v1.train.GradientDescentOptimizer(1);

train = gra_op.minimize(loss_value);

init = tf.compat.v1.global_variables_initializer();

sess = tf.compat.v1.Session();
sess.run(init);

for i in range(10000000):
    sess.run(train, {x: X_train, y: Y_train});

cW, cb, c_loss = sess.run([W, b, loss_value], {x: X_train, y: Y_train});

print("W: %s %s, b: %s, loss: %s"%(cW[0], cW[1], cb, c_loss));

# y = 2/3 x + 1/3, [7, 5], [-8, -5]
plt.scatter(X_train[:,0], X_train[:,1], c = Y_train);
x = np.linspace(-15, 15, 1000000);
y = 2.0/3.0 * x + 1.0/3.0
plt.plot(x, y);
y_new = list(cW[0])[0]/list(cW[1])[0] * x + float(cb)/list(cW[1])[0];
plt.plot(x, y_new);
plt.show();