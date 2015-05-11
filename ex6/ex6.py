import numpy as np


def perceptron(data, labels):
    num_samples, pixels = data.shape
    w = np.zeros(pixels)

    found_miss = True

    while found_miss:
        found_miss = False

        for i in range(0, num_samples):
            if labels[i] * np.inner(w, data[i]) <= 0:
                w += data[i] * labels[i]
                found_miss = True

    return w


def compute_error(data, labels, w):
    num_samples, pixels = data.shape
    total_error = 0.0

    for i in range(0, num_samples):
        if labels[i] * np.inner(w, data[i]) <= 0:
            total_error += 1

    return total_error / num_samples


if __name__ == '__main__':
    xTrain = np.loadtxt("Xtrain");
    yTrain = np.loadtxt("Ytrain");
    xTest = np.loadtxt("Xtest");
    yTest = np.loadtxt("Ytest");

    trainResult = perceptron(xTrain, yTrain)
    avgError = compute_error(xTest, yTest, trainResult)

    print("Average loss (0-1): {0:.4f}".format(avgError))

