__author__ = 'jirsch'

import numpy as np
import numpy.matlib as mlib
import random
import svm_utils as su
import matplotlib.pyplot as plt


def linear_svm(X, Y, lamb, T):
    num_of_samples, dim = X.shape
    theta = np.zeros((1, dim))
    sum_w = np.zeros((1, dim))

    for t in range(1, T):
        w = np.divide(theta, lamb * t)
        i = random.randint(0, num_of_samples - 1)
        inner = np.dot(w, X[i, :])

        if Y[i] * inner < 1:
            theta += np.multiply(X[i, :], Y[i])

        sum_w += w

    sum_w = np.divide(sum_w, T)
    return np.transpose(sum_w)


def gaussian_kernel_svm(X, Y, lamb, sigma2, T):
    num_of_samples, dim = X.shape
    Z = np.dot(X, np.transpose(X))
    v = np.diag(Z)
    D = mlib.repmat(v, num_of_samples,1)

    exp = np.divide(D + np.transpose(D) - np.multiply(Z,2), 2 * sigma2)
    G = np.exp(-exp)


    beta = np.zeros((num_of_samples))
    alpha = np.zeros((num_of_samples))
    sum_alpha = np.zeros((num_of_samples))

    for t in range(1, T):
        alpha = np.divide(beta, lamb * t)
        i = random.randint(0, num_of_samples - 1)

        kernel_sum = np.dot(G[:,i],alpha)

        if Y[i] * kernel_sum < 1:
            beta[i] += Y[i]

        sum_alpha += alpha

    sum_alpha = np.divide(sum_alpha, T)
    return sum_alpha


if __name__ == '__main__':
    xLinear = np.loadtxt("X_Linear")
    xGaussian = np.loadtxt("X_gaussian")
    yLinear = np.loadtxt("Y_Linear")
    yGaussian = np.loadtxt("Y_gaussian")

    m = xLinear.shape[0]
    lin_result = linear_svm(xLinear, yLinear, 0.01, 10 * m)
    su.show_SVM_linear(xLinear, yLinear, lin_result)
    plt.show()

    for sigma in {10,1,0.1}:
        print(sigma)
        gauss_result = gaussian_kernel_svm(xGaussian,yGaussian,0.1,sigma,10*m)
        su.show_SVM_gaussian(xGaussian,yGaussian,gauss_result,sigma)
        plt.show()


