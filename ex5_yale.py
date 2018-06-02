import random

import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import os
import numpy as np


def load_data(path):
    list_dir = os.listdir(path)
    X = []
    for img in list_dir:
        if img == "Readme.txt":
            continue
        else:
            im = plt.imread(path + img)
            X.append(im.flatten())
    return np.matrix(X)


def show_image(X):
    image = X[random.randint(0, 165)]
    image = image.reshape((243, 320))
    plt.imshow(image, cmap='gray')
    plt.show()


def main():
    # task a
    X = load_data("data/yalefaces/")

    # task b
    m = X.mean(1)
    X_centered = X - m

    # task c
    u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    # u, s, vt = svds(X_centered)

    # task d
    p = 60
    V_p = vt.T[:, 0:p]
    Z = np.dot(X_centered, V_p)

    # task e
    X_rec = m + np.dot(Z, V_p.T)
    error = np.square(np.subtract(X_rec, X)).sum()

    print "Reconstraction error:", error

    show_image(X_rec)


if __name__ == '__main__':
    main()