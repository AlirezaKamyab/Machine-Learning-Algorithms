# This file is written by Alireza Kamyab
# Copy right 2022
import numpy as np
import copy

# z either a scale or n-dimensional array 
def segmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Function of Logistic Regression used to predict
# x is a vector or values for each feature
# w is a vector of the weight for each feature
# b is a scalar for bias
def predict(x, w, b):
    return segmoid(np.dot(x, w) + b)


# Calculates the loss compared to the actual value
# fwb_i is the result of the prediction
# y is the actual value
def loss(fwb_i, y):
    return -y * np.log(fwb_i) - (1 - y) * np.log(1 - fwb_i)


# Cost function returns the average loss plus regularization term
# X is a matrix consisting of m rows (entries) and n columns (features)
# y is the vector of training outputs
# w is a vector of the weight for each feature
# b is a scalar for bias
# lambda is regularization parameter
def cost(X, y, w, b, lambda_ = 0):
    m, n = X.shape
    loss_sum = 0
    for i in range(m):
        loss_sum += loss(predict(X[i], w, b), y[i])
    total_cost = loss_sum / m
    
    reg_sum = 0
    for j in range(n):
        reg_sum += w[j]
    total_cost += (lambda_ / (2 * m)) * reg_sum
    return total_cost


# X is a matrix consisting of m rows (entries) and n columns (features)
# y is the vector of training outputs
# w is a vector of the weight for each feature
# b is a scalar for bias
# dj_dw a vector which is the derivative of the cost function with respect to each feature
# dj_db a scalar which is the derivative of the cost function with respect to b
# lambda is regularization parameter
def gradient(X, y, w, b, lambda_ = 0):
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
        dj_dw[j] += (lambda_ / m) * w[j]
    
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
                        lambda_ = 0, print_progress=True):
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
# y is the vector of training outputs
# w is a vector of the weight for each feature
# b is a scalar for bias
# threshold is a scalar by which decision is made - res >= threshold returns 1 else returns 0
def calc_accuracy(X, y, w_out, b_out, threshold=0.5):
    correct = 0
    m, _ = X.shape
    for i in range(m):
        prediction = predict(X[i], w_out, b_out)
        if prediction >= threshold and y[i] == 1: correct += 1
        elif prediction < threshold and y[i] == 0: correct += 1
    return correct / m


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


# Returns data to plot a contourf plot
# X is a matrix consisting of m rows (entries) and n columns (features)
# points is a scalar that makes a grid with each point step
def plot_data(X, points = 0.1):
    min1, max1 = X[:, 0].min() - 1, X[:, 1].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1grid = np.arange(min1, max1, points)
    x2grid = np.arange(min2, max2, points)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = np.reshape(r1, (-1, 1)), np.reshape(r2, (-1, 1))
    x_grid = np.hstack((r1, r2))
    
    return xx, yy, x_grid