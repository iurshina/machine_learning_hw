import matplotlib.pyplot as plt
import os
import math
import numpy as np
from collections import defaultdict


def load_data(path):
    list_dir = os.listdir(path)
    X = []
    H, W = 0, 0
    for img in list_dir:
        if img == "Readme.txt":
            continue
        else:
            im = plt.imread(path + img)
            H, W = np.shape(im)
            X.append(im.flatten())

    return np.matrix(X), H, W


def show_image(image, H, W):
    image = image.reshape((H, W))
    plt.imshow(image, cmap='gray')
    plt.show()


def image_grid(D, H, W, title, cols=10, scale=1):
    n = np.shape(D)[0]
    rows = int(math.ceil((n + 0.0) / cols))
    plt.figure(1, figsize=[scale * 20.0 / H * W, scale * 20.0 / cols * rows], dpi=300)
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.reshape(D[i, :], [H, W]), cmap=plt.get_cmap("gray"))
        plt.axis('off')
    plt.title(title)
    plt.show()


def k_means(X, k, steps):
    initial_m_idx = np.random.randint(len(X), size=k)
    m = X[initial_m_idx]

    clusters = defaultdict(list)
    total_error = 0
    for s in range(0, steps):
        clusters.clear()
        total_error = 0
        for j in range(0, len(X)):
            min_dist = np.linalg.norm(X[j] - m[0], 2)
            cluster_id = 0
            for i in range(1, len(m)):
                cur_dist = np.linalg.norm(X[j] - m[i], 2)
                if min_dist > cur_dist:
                    min_dist = cur_dist
                    cluster_id = i
            clusters[cluster_id].append(j)
            total_error += min_dist

        print str(s) + ": " + str(total_error)

        if s == steps - 1:
            break

        for k in clusters:
            m[k] = X[clusters[k]].mean(axis=0)

    return m, total_error, clusters


def main():
    X, H, W = load_data("data/yalefaces/")
    # task a
    k = 4
    steps = 10

    m, error, clusters = k_means(X, k, steps)
    image_grid(m, H, W, "Means for task a")

    # for key in clusters:
        # image_grid(X[clusters[key]], H, W, "Cluster " + str(key) + " for task a")

    # # task b
    ks = [2, 3, 4, 5, 6, 7, 9, 10, 15]
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

    means, error, clusters = k_means(Z, k, steps)
    # show_image(m + np.dot(means[0], V_p.T), H, W)

    # image_grid([m + np.dot(mm, V_p.T) for mm in means], H, W, "Means for task c")

    for key in clusters:
        image_grid(X[clusters[key]], H, W, "Cluster " + str(key + 1) + " for task c")

    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    errors = []
    for k in ks:
        errors.append(k_means(Z, k, steps)[1])

    plt.plot(ks, errors)
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()


if __name__ == '__main__':
    main()