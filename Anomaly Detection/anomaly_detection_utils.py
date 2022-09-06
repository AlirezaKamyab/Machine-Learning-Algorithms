# This file is written by Alireza Kamyab;

import numpy as np

# Calculates mean and variance for gaussian distribution
# X ndarray - Matrix with the shape of (m, n) with m data and n features
# returns mean and variance
def estimate_gaussian(X):
    m, _ = X.shape
    mu = X.sum(axis=0) / m
    var = ((X - mu) ** 2).sum(axis=0) / m
    return mu, var


# Calculates gaussian probabily
# X ndarray - Matrix with the shape of (m, n) with m data and n features
# mu vector - Shape (1, n), which are the mean for n features
# var vector - Shape (1, n), which are the variance for n features
# returns mean and variance
def gaussian_probability(X, mu, var):
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p


# Finds best epsilon given a dataset of probabilities and evaluated results
# p_val vector - Shape(1, m), containing the gaussian probability of each data
# y_val vector - Shape(1, m), containing classification results of being anomaly or normal
# returns best_epsilon and F1 score
def select_threshold(y_val, p_val): 
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        p_res = p_val < epsilon
        tp = np.sum((p_res == 1) & (y_val == 1))
        fp = np.sum((p_res == 1) & (y_val == 0))
        fn = np.sum((p_res == 0) & (y_val == 1))
        
        precision = tp / (tp + fp)
        recall =  tp / (tp + fn)
        
        F1 = (2 * precision * recall) / (precision + recall)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1


