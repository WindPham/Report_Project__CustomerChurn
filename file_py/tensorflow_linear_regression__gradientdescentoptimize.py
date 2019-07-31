
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
    return (1.0/3.0)*x - 8;
def generate_data(N):
    X_train = np.random.uniform(-10.0, 10.0 , size = N);
    #print(X_train);
    rd = np.random.uniform(10 , 20 , size = 2)
    Y_train = fun(X_train) + X_train/rd[0] - X_train/rd[1];# create small noise
    print(fun(X_train)[0:5])
    #print(Y_train);
    return X_train, Y_train;

X_train, Y_train = generate_data(10000);
print(X_train[0:5])
print(Y_train[0:5])
#target: y = x/3 - 8
W = tf.Variable([1],dtype = tf.float32);
b = tf.Variable([1],dtype = tf.float32);
X = tf.compat.v1.placeholder(tf.float32);
Y = tf.compat.v1.placeholder(tf.float32);
#khai bao mo hinh
linear_model = W*X + b;
#khai bao ham mat mac
loss_value = tf.reduce_sum(tf.square(linear_model - Y));
#khai bao ham toi uu ham mat mac
gradient_op = tf.compat.v1.train.GradientDescentOptimizer(0.000001);
train =  gradient_op.minimize(loss_value);
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init) # reset values to wrong
#vong for lap 1000 lan
for i in range(1000):
    sess.run(train, {X:X_train, Y:Y_train})
#in ket qua
curr_W, curr_b, curr_loss = sess.run([W, b, loss_value], {X:X_train, Y:Y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
x = np.linspace(-10, 10, 4000);
y = curr_W*x + curr_b;
plt.scatter(x = X_train, y = Y_train, alpha = 0.3);
plt.plot(x, y, 'red')
plt.plot(x, fun(x), 'black', alpha = 1)
plt.show();
