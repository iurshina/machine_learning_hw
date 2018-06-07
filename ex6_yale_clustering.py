import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict


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


def show_image(image):
    image = image.reshape((243, 320))
    plt.imshow(image, cmap='gray')
    plt.show()


def k_means(X, k, steps):
    initial_m_idx = np.random.randint(len(X), size=k)
    m = X[initial_m_idx, :]

    clusters = defaultdict(list)
    total_error = 0
    for s in range(0, steps):
        clusters.clear()
        total_error = 0
        for j in range(0, len(X)):
            min_er = np.square(np.subtract(X[j], m[0])).sum()
            cluster_id = 0
            for i in range(1, len(m)):
                cur_er = np.square(np.subtract(X[j], m[i])).sum()
                if min_er > cur_er:
                    min_er = cur_er
                    cluster_id = i
            clusters[cluster_id].append(j)
            total_error += min_er

        print str(s) + ": " + str(total_error)

        if s == steps - 1:
            break

        for k in clusters:
            m[k] = np.mean(X[clusters[k], :], axis=0)

    return m, total_error


def main():
    X = load_data("data/yalefaces/")
    # task a
    k = 4
    steps = 10

    m, error = k_means(X, k, steps)
    show_image(m[0])

    # task b
    ks = [2, 4, 8, 16, 32, 64]
    errors = []
    for k in ks:
        errors.append(k_means(X, k, steps)[1])

    plt.plot(ks, errors)
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()

    # task c (with PCA it converges)
    m = X.mean(axis=0)
    X_centered = X - m

    u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    p = 20
    V_p = vt.T[:, 0:p]
    Z = np.dot(X_centered, V_p)

    k = 4

    means, error = k_means(Z, k, steps)
    show_image(m + np.dot(means[0], V_p.T))

    ks = [2, 4, 8, 16, 32, 64]
    errors = []
    for k in ks:
        errors.append(k_means(Z, k, steps)[1])

    plt.plot(ks, errors)
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()


if __name__ == '__main__':
    main()