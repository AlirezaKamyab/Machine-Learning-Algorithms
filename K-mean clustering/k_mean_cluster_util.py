#!/usr/bin/env python3
# This file is written by Alireza Kamyab;

import numpy as np

# Find L2 norm of x
def norm(x):
    return np.linalg.norm(x)


# Calculates distances from each points to its assigned centroid 
# X ndarray - Matrix of data with m rows and n columns which are data and features respectively
# idx vector - 1 row and m columns; idx[i] represents index of centeroid which X[i] belongs to
# centeroids vector - n row and K columns; shows position of each centeroid
# returns a vector of distances^2
def calc_distances_to_centroids(X, idx, centroids):
    distances = np.zeros(X.shape[0])
    m,_ = X.shape
    for i in range(m):
        distances[i] = norm(X[i] - centroids[idx[i]]) ** 2
    return distances


# Calculates Distortion Cost
# X ndarray - Matrix of data with m rows and n columns which are data and features respectively
# idx vector - 1 row and m columns; idx[i] represents index of centeroid which X[i] belongs to
# centeroids vector - n row and K columns; shows position of each centeroid
# returns mean of distances^2
def cost(X, idx, centroids):
    summation = calc_distances_to_centroids(X, idx, centroids)
    summation = np.sum(summation)
    return summation / X.shape[0]


# Assign each point X[i] to its closest cluster centeroid
# X ndarray - Matrix of data with m rows and n columns which are data and features respectively
# centeroids vector - n row and K columns; shows position of each centeroid
# returns idx with 1 row and m columns; idx[i] represents index of centeroid which X[i] belongs to
def assign_centroids(X, centroids):
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.zeros(centroids.shape[0])
        for k in range(centroids.shape[0]):
            distances[k] = norm(X[i] - centroids[k])
        idx[i] = np.argmin(distances)
    return idx


# Move each centroid to the mean of points which are assigned to that centeroid
# X ndarray - Matrix of data with m rows and n columns which are data and features respectively
# idx vector - 1 row and m columns; idx[i] represents index of centeroid which X[i] belongs to
# centeroids vector - n row and K columns; shows position of each centeroid
# returns new centeroid
def move_centroids(X, idx, centroids):
    m, n = X.shape
    K = centroids.shape[0]
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)
    return centroids


# Runs k_mean algorithm to find clusters in the data
# X ndarray - Matrix of data with m rows and n columns which are data and features respectively
# initial_centeroids vector - n row and K columns; initial values for each centroid
# returns centeroid and idx
def k_mean(X, initial_centroids, max_iter):
    centroids = initial_centroids
    K = initial_centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype='int')
    for i in range(1, max_iter + 1):
        print(f'Iteration {i}/{max_iter}')
        idx = assign_centroids(X, centroids)
        centroids = move_centroids(X, idx, centroids)
        print(f'Cost: {cost(X, idx, centroids)}')
    return centroids, idx


# Randomly chooses K points to be initial centeroids
# X ndarray - Matrix of data with m rows and n columns which are data and features respectively
# K int - The number of clusters
# returns vector with 1 row and K columns and initial centeroid
def random_initial_centroids(X, K):
    a = np.random.permutation(X.shape[0])
    return X[a[:K]]