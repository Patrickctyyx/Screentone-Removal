import numpy as np


def gauss(x, mean, sigma):
    return np.exp((-(x - mean) * (x - mean)) / (2 * sigma * sigma)) / np.sqrt(np.pi * 2 * sigma * sigma)


def make_gauss_vector(sigma):
    threshold = 0.001

    i = 0
    while 1:
        i += 1
        if gauss(i, 0, sigma) < threshold:
            break

    gau = [0] * (i + 1)
    for j in range(1, len(gau)):
        gau[j] = gauss(j, 0, sigma)

    return gau
