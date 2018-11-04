
import sys, os
import numpy as np
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x)) 

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0] 
    return -np.sum(np.multiply(np.log(y + 1e-7), t)) / batch_size 

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad 


def predict(ts, _w1, _b1, _w2, _b2):

    A = np.dot(ts, _w1) + _b1 
    z = sigmoid(A)
    B = np.dot(z, _w2) + _b2
    y = softmax(B)
    return y

def loss(ts, _w1, _b1, _w2, _b2, t):
    A = predict(ts, _w1, _b1, _w2, _b2) 
    return cross_entropy_error(A, t)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


w1 = 0.01 * np.random.randn(784, 100)
b1 = np.zeros(100)
w2 = 0.01 * np.random.randn(100, 10) 
b2 = np.zeros(10)


for i in range(3):
    batch_mask = np.random.choice(50, 100)
    trainSet = x_train[batch_mask]
    lossW = lambda W: loss(trainSet, w1, b1, w2, b2, t_train[batch_mask])
    w1Grad = numerical_gradient(lossW, w1)
    b1Grad = numerical_gradient(lossW, b1)
    w2Grad = numerical_gradient(lossW, w2)
    b2Grad = numerical_gradient(lossW, b2)

    w1 = w1 - 0.1 * w1Grad
    b1 = b1 - 0.1 * b1Grad
    w2 = w2 - 0.1 * w2Grad
    b2 = b2 - 0.1 * b2Grad

    print(lossW(0))

for i in range(1):
    batch_mask = np.random.choice(50, 10)
    testX = x_train[batch_mask]
    testY = t_train[batch_mask]
    pre = predict(testX, w1, b1, w2, b2)
    print(pre, testY)
    print(np.argmax(pre, axis=1), np.argmax(testY, axis=1))
    print("---------")
# b1Grad = numerical_gradient(loss, b1)
# w2Grad = numerical_gradient(loss, w2)
# b2Grad = numerical_gradient(loss, b2)

# print(t_train[batch_mask])
# P = predict(trainSet, w1, b1, w2, b2, t_train[batch_mask])
# print(P)



