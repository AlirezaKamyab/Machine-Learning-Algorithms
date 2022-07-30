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
# lambda is regularization parameter
def cost(X, y, w, b, lambda_ = 0):
    m, n = X.shape
    cost_sum = 0
    for i in range(m):
        err = predict(X[i], w, b) - y[i]
        cost_sum += err ** 2
    cost_sum = cost_sum / (2 * m)
    
    reg_sum = 0
    for j in range(n):
        reg_sum += w[j] ** 2
    reg_sum = reg_sum * (lambda_ / (2 * m))
    
    return cost_sum + reg_sum


# X is a matrix consisting of m rows (entries) and n columns (features)
# y is the vector of training outputs
# w is a vector of the weight for each feature
# b is a scalar for bias
# dj_dw a vector which is the derivative of the cost function with respect to each feature
# dj_db a scalar which is the derivative of the cost function with respect to b
# lambda is regularization parameter
def gradient(X, y, w, b, lambda_= 0):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    
    for i in range(m):
        err = predict(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i][j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]
    
    return dj_dw, dj_db


# X is a matrix consisting of m rows (entries) and n columns (features)
# y is the vector of training outputs
# w is a vector of the weight for each feature as initial value
# b is a scalar for bias as initial value
# alpha scalar, learning rate
# iterations, number of updates that should be made
# hist, history of every update
# lambda is regularization parameter
# print_progress is a boolean, if true, prints w and b and the cost on each 1/10 of the steps
def gradient_descent(X, y, init_w, init_b, alpha, cost_function, gradient_function, iterations = 1000,
                     lambda_ = 0, print_progress = True):
    w = copy.deepcopy(init_w)
    b = init_b
    hist = []
    hc = 10 ** max([0, np.log10(iterations) - 1])
    
    for i in range(1, iterations + 1):
        dj_dw, dj_db = gradient_function(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i % hc == 0:
            calced_cost = cost_function(X, y, w, b)
            hist.append([i, w, b, calced_cost])
            if print_progress: print(f'i: {i} w={w} b={b} cost={calced_cost}')
    return w, b, hist


# X is a matrix consisting of m rows (entries) and n columns (features)
def z_normalization(X):
    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    x_norm = (X - mu) / sigma
    return x_norm, mu, sigma


# X is a matrix consisting of m rows (entries) and n columns (features)
def mean_normalization(X):
    mu = np.mean(X, axis=0)
    x_mean = X / mu
    return x_mean, mu


# X is a matrix consisting of m rows (entries) and n columns (features)
def scale(X):
    abs_max = np.max(np.abs(X), axis=0)
    x_scaled = X / abs_max
    return x_scaled, abs_max