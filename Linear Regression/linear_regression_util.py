# This file is written by Alireza Kamyab
# Copy right 2022
import numpy as np
import copy

# This function returns the predicted function 
# x is a vector or values for each feature
# w is a vector of the weight for each feature
# b is a scalar for bias
def predict(x, w, b):
    return np.dot(x, w) + b


# Cost function returns the average squared error / 2
# X is a matrix consisting of m rows (entries) and n columns (features)
# y is the vector of training outputs
# w is a vector of the weight for each feature
# b is a scalar for bias
def cost(X, y, w, b):
    m, _ = X.shape
    sq_err_sum = 0
    for i in range(m):
        sq_err_sum += (predict(X[i], w, b) - y[i]) ** 2
    return sq_err_sum / (2 * m)


# X is a matrix consisting of m rows (entries) and n columns (features)
# y is the vector of training outputs
# w is a vector of the weight for each feature
# b is a scalar for bias
# dj_dw a vector which is the derivative of the cost function with respect to each feature
# dj_db a scalar which is the derivative of the cost function with respect to b
def gradient(X, y, w, b):
    dj_dw = np.zeros(n)
    dj_db = 0
    m, n = X.shape
    for i in range(m):
        err = predict(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (err * X[i][j])
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


# X is a matrix consisting of m rows (entries) and n columns (features)
# y is the vector of training outputs
# w is a vector of the weight for each feature as initial value
# b is a scalar for bias as initial value
# alpha scalar, learning rate
# iterations, number of updates that should be made
# hist, history of every update
def gradient_descent(X, y, init_w, init_b, alpha, iterations=1000):
    w = copy.deepcopy(init_w)
    b = init_b
    hist = []
    for i in range(1, iterations):
        dj_dw, dj_db = gradient(X, y, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        hist.append[[i, w, b, cost(X, y, w, b)]]
    return w, b, np.array(hist)


# X is a matrix consisting of m rows (entries) and n columns (features)
def z_normalization(X):
    x_norm = copy.deepcopy(X)
    m, n = x_norm.T.shape
    for i in range(m):
        std = x_norm[i].std()
        mean = x_norm[i].mean()
        x_norm[i] = (x_norm[i] - mean) / std
    return x_norm.T


# X is a matrix consisting of m rows (entries) and n columns (features)
def mean_normalization(X):
    x_norm = copy.deepcopy(X)
    m, n = x_norm.T.shape
    for i in range(m):
        mean = x_norm[i].mean()
        x_norm[i] = x_norm[i] / mean
    return x_norm.T


# X is a matrix consisting of m rows (entries) and n columns (features)
def scale(X):
    x_norm = copy.deepcopy(X)
    m, n = x_norm.T.shape
    for i in range(m):
        max_value = x_norm[i].max()
        x_norm[i] = x_norm[i] / max_value
    return x_norm.T