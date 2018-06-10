import random

import numpy as np


def load_data(file_name):
    data = np.loadtxt(file_name)
    return data


def G(x, cov, m, k):
    # is sum ok?
    return 1. / (((2 * np.pi) ** k) * np.linalg.det(cov)) * np.exp(-.5 * np.sum((x - m).T * np.linalg.inv(cov) * (x - m)).T)


def gmm_initialized_means(X, k, steps):
    n, d = X.shape

    # probabilities for each Gaussian (weights)
    pi = [1. / k] * k
    # covariance matrices
    cov = [np.identity(d)] * k

    # initial means
    initial_m_idx = np.random.randint(len(X), size=k)
    m = X[initial_m_idx, :]

    membership_probs = np.zeros((n, k))
    log_likelihoods = []
    for i in range(steps):
        # E-step
        for j in range(k):
            # probabilities of each of the data points belonging to a Gaussian
            for i in range(0, n):
                membership_probs[i, j] = pi[j] * G(X[i], cov[j], m[j], k)

            log_likelihoods.append(np.sum(np.log(np.sum(membership_probs, axis=1))))

            # normalize
            membership_probs = (membership_probs.T / np.sum(membership_probs, axis=1)).T

        # M-step
        for j in range(k):
            n_k = np.sum(membership_probs, axis=0)

            # update parameters
            pi[j] = 1. / n * n_k[j]
            m[j] = 1. / n_k[j] * np.sum(membership_probs[:, j] * X.T, axis=1).T
            cov[j] = np.array(1. / n_k[j] * np.dot(np.multiply(np.matrix(X - m[j]).T, membership_probs[:, j]), np.matrix(X - m[j])))

        if len(log_likelihoods) > 2 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < 0.0001:
            print "mean-initialized converged"
            break


def gmm_initialized_posterior(X, k, steps):
    n, d = X.shape

    pi = [0] * k
    cov = [np.zeros(d)] * k
    m = [np.zeros(d)] * k

    membership_probs = np.zeros((n, k))
    for i in range(0, n):
        k_i = random.randint(0, k - 1)
        membership_probs[i, k_i] = 1

    log_likelihoods = []
    for i in range(steps):
        # M-step
        for j in range(k):
            n_k = np.sum(membership_probs, axis=0)

            pi[j] = 1. / n * n_k[j]
            m[j] = 1. / n_k[j] * np.sum(membership_probs[:, j] * X.T, axis=1).T
            cov[j] = np.array(1. / n_k[j] * np.dot(np.multiply(np.matrix(X - m[j]).T, membership_probs[:, j]), np.matrix(X - m[j])))

        # E-step
        for j in range(k):
            for i in range(0, n):
                membership_probs[i, j] = pi[j] * G(X[i], cov[j], m[j], k)

            log_likelihoods.append(np.sum(np.log(np.sum(membership_probs, axis=1))))

            membership_probs = (membership_probs.T / np.sum(membership_probs, axis=1)).T

        if len(log_likelihoods) > 2 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < 0.0001:
            print "posterior-initialized converged"
            break


def main():
    X = load_data("data/mixture.txt")
    k = 3
    steps = 1000

    # task a
    gmm_initialized_means(X, k, steps)

    # task b
    gmm_initialized_posterior(X, k, steps)


if __name__ == '__main__':
    main()