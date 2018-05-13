#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def mdot(*args):
    return reduce(np.dot, args)


def prepend_one(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def prepend_quadratic_features(X):
    ones = np.ones((X.shape[0], 1))
    X_sq = X**2
    return np.column_stack((ones, X, X_sq[:, 0], X_sq[:, 1], X[:, 0] * X[:, 1]))


def load_data(file_name):
    data = np.loadtxt(file_name)
    return data


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient(X, y, beta, lm):
    p = sigmoid(np.dot(X, beta))

    return np.dot(X.T, np.subtract(p, y)) + 2 * lm * np.dot(np.identity(X.shape[1]), beta)


def hessian(X, beta, lm):
    p = sigmoid(np.dot(X, beta))

    W = np.diag(p * (1 - p))

    return X.T.dot(W).dot(X) + 2 * lm * np.identity(X.shape[1])


def newton_method(X, y, lm, n):
    beta = np.zeros((X.shape[1]))

    for i in range(n):
        H = hessian(X, beta, lm)
        G = gradient(X, y, beta, lm)

        beta -= np.dot(np.linalg.inv(H), G)

    return beta


def plot_data(data):
    ones = np.array(data[:, -1] == 1)

    plt.plot(data[ones, 0], data[ones, 1], 'ro')
    plt.plot(data[~ones, 0], data[~ones, 1], 'bv')


def plot_res_with_prob(data, p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ones = np.array(data[:, -1] == 1)
    ones = np.array(p[:] > 0.5)

    ax.scatter(data[ones, 0], data[ones, 1], p[ones], c='r', marker='o')
    ax.scatter(data[~ones, 0], data[~ones, 1], p[~ones], c='b', marker='v')

    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('p')


def task_a(lm, steps):
    data = load_data("data2Class.txt")
    X = data[:, :-1]
    y = data[:, -1]
    X = prepend_one(X)

    beta = newton_method(X, y, lm, steps)

    print "Optimum beta (linear): \n", beta

    p = sigmoid(np.dot(X, beta))

    plot_data(data)
    plot_res_with_prob(data, p)

    plt.show()


def task_b(lm, steps):
    data = load_data("data2Class.txt")
    X = data[:, :-1]
    y = data[:, -1]
    X = prepend_quadratic_features(X)

    beta = newton_method(X, y, lm, steps)

    print "Optimum beta (quadratic): \n", beta

    p = sigmoid(np.dot(X, beta))

    # plot_data(data)
    plot_res_with_prob(data, p)

    plt.show()


def main():
    lm = 0.00001
    steps = 10

    task_a(lm, steps)
    task_b(lm, steps)


if __name__ == '__main__':
    main()

