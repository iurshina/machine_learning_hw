from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


alpha = .05


def load_data(file_name):
    data = np.loadtxt(file_name)
    return data


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward(x, W_h, W_o):
    layer = sigmoid(np.dot(x, W_h))
    predicted = np.dot(layer, W_o)

    return layer, predicted.ravel()


def backward(y, predicted, hidden, W_o):
    delta_o = -y * (1.- y * predicted)
    delta_h = hidden * (1. - hidden) * W_o.ravel() * delta_o

    return delta_h, delta_o


def hinge_loss(actual, expected):
    v = 1. - actual * expected
    v[v < 0.] = 0
    return v.mean()


def update_weights(x, hidden, W_h, W_o, delta_h, delta_o):
    update_o = -alpha * hidden * delta_o
    update_h = -alpha * np.dot(x.reshape((-1, 1)), delta_h.reshape((1, -1)))

    W_o += update_o.reshape((-1, 1))
    W_h += update_h

    return W_h, W_o


def plot(data, f):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ones = np.array(f[:] > 0)

    ax.scatter(data[ones, 1], data[ones, 2], f[ones], c='r', marker='o')
    ax.scatter(data[~ones, 1], data[~ones, 2], f[~ones], c='b', marker='v')

    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('f')

    plt.show()


def main():
    data = load_data("data/data2Class_adjusted.txt")

    X, Y = data[:, :-1], data[:, -1]

    features_n = X.shape[1]
    hidden_n_n = 100
    n_classes = 1

    W_h = np.random.random((features_n, hidden_n_n))
    W_o = np.random.random((hidden_n_n, n_classes))

    N = 1000
    for s in range(N):
        f = []
        hidden, predicted = forward(X, W_h, W_o)
        loss = hinge_loss(Y, predicted)
        num_err = (np.sign(predicted) != Y).sum()

        for x, y in zip(X, Y):
            hidden, predicted = forward(x, W_h, W_o)
            delta_h, delta_o = backward(y, predicted, hidden, W_o)

            W_h, W_o = update_weights(x, hidden, W_h, W_o, delta_h, delta_o)

            f.append(predicted)

        print ("Iteration: ", s, " loss: ", loss, " num_err: ", num_err)

    plot(data, np.array(f).ravel())


if __name__ == '__main__':
    main()