#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import dot
from numpy.linalg import inv
from mpl_toolkits.mplot3d.axes3d import Axes3D


def mdot(*args):   
    return reduce(np.dot, args)


def prepend_one(X):
    # first element to zero
    return np.column_stack([np.ones(X.shape[0]), X])


def prepend_quadratic_features(X):
    ones = np.ones((X.shape[0], 1))
    X_sq = X**2
    return np.column_stack((ones, X, X_sq[:, 0], X_sq[:, 1], X[:, 0] * X[:, 1]))    


def load_data(file_name):
    data = np.loadtxt(file_name)    
    return data
    
    
def compute_mse(y, test_y):
    errors = (y - test_y)**2
    return np.mean(errors), errors
    

class RidgeEstimator(object):
    def __init__(self, lm):
        self.lm = lm
        
    def fit(self, X, y):
        self.beta = mdot(inv(dot(X.T, X) + self.lm * np.identity(X.shape[1])), X.T, y)

    def predict(self, X):       
        return mdot(X, self.beta)


def cross_validation(X, y, k, est):
    N = X.shape[0]
    indexes = np.arange(N)
    block_size = N // k
    
    errors = []
    for i in range(0, k):
        s, e = i * block_size, min((i + 1) * block_size, N)
                
        train_indexes = np.hstack((indexes[:s], indexes[e:]))
        test_indexes = indexes[s:e]
        X_train, y_train = X[train_indexes], y[train_indexes]
        X_test, y_test = X[test_indexes], y[test_indexes]
                                                          
        est.fit(X_train, y_train)        
        errors.append(compute_mse(est.predict(X_test), y_test)[0])        
    
    return np.mean(errors)    


def task_a(lm):
    data = load_data("dataLinReg2D.txt")
    X = data[:, :-1]
    y = data[:, -1]
    X = prepend_one(X)
    
    est = RidgeEstimator(lm)
    est.fit(X, y)
    mse, errors = compute_mse(est.predict(X), y)
    
    print('Task A# (lambda=%g): MSE=%.4f' % (lm, mse))
    return errors


def task_b(lm):
    data = load_data("dataQuadReg2D.txt")
    X = data[:, :-1]
    y = data[:, -1]
    X = prepend_quadratic_features(X)

    est = RidgeEstimator(lm)    
    est.fit(X, y)
    mse, errors = compute_mse(est.predict(X), y)
    
    print('Task B# (lambda=%g): MSE=%.4f' % (lm, mse))
    return errors


def task_c(lm):
    data = load_data("dataQuadReg2D_noisy.txt")
    X = data[:, :-1]
    y = data[:, -1]
    X = prepend_quadratic_features(X)

    est = RidgeEstimator(lm)        
    est.fit(X, y)
    mse, errors = compute_mse(est.predict(X), y)
    mse_cv = cross_validation(X, y, 5, est)
    
    print('Task C# (lambda=%g): MSE=%.4f, mean MSE from CV=%.4f' % (lm, mse, mse_cv))
    return errors, mse_cv


def main():    
    lambdas = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 500, 1000]
    
    for task in [task_a, task_b]:
        errors = [(lm, error) for lm in lambdas for error in task(lm)]
        
        df = pd.DataFrame(errors,  columns=['$\lambda$', 'MSE'])
        sns.barplot(x='$\lambda$', y='MSE', data=df)                 
        plt.title('MSE for %s' % task.__name__)        
        plt.xticks(rotation=45)
        plt.show()
                
    c_results = []
    for lm in lambdas:
        errors, mse_cv = task_c(lm)
        c_results.append((lm, mse_cv, 'MSE for CV'))
        for error in errors:
            c_results.append((lm, error, 'MSE'))        
    df = pd.DataFrame(c_results, columns=[r'$\lambda$', 'MSE', 'Data'])
    sns.barplot(x=r'$\lambda$', y='MSE', hue='Data', data=df)                 
    plt.title('MSE for task_c')
    plt.xticks(rotation=45)
    plt.show()
                
        
if __name__ == '__main__':
    main()        