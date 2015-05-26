__author__ = 'jirsch'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def least_squares(data, labels, degree):
    numSamples = data.shape[0]
    xPoly = np.zeros((degree + 1, numSamples))

    for i in range(0, numSamples):
        for j in range(0, degree + 1):
            xPoly[j, i] = data[i] ** j

    X = np.dot(xPoly, np.transpose(xPoly))

    xDagger = np.linalg.pinv(X)

    classifier = np.dot(np.dot(xDagger, xPoly), labels)
    error = calc_error(classifier, data, labels)

    return (classifier, error)


def calc_error(classifier, data, labels):
    num_samples = data.shape[0]
    elements = classifier.shape[0]

    error = 0

    for sample in range(0, num_samples):
        inner = 0
        for element in range(0, elements):
            inner += classifier[element] * (data[sample] ** element)

        error += ((inner - labels[sample]) ** 2) / num_samples

    return error


def validate(classifiers, data, labels):
    minIdx = 0
    minError = calc_error(classifiers[0], data, labels)

    num_of_classifiers = len(classifiers)

    validation_errors = np.zeros((num_of_classifiers, 2))

    for i in range(0, num_of_classifiers):
        currError = calc_error(classifiers[i], data, labels)
        validation_errors[i] = (i + 1, currError)

        if currError < minError:
            minError = currError
            minIdx = i

    return minIdx, validation_errors


if __name__ == '__main__':
    max_deg = 15

    xAll = np.loadtxt("X.txt")
    yAll = np.loadtxt("Y.txt")

    xTrain = xAll[0:20]
    yTrain = yAll[0:20]

    xValidate = xAll[20:121]
    yValidate = yAll[20:121]

    xTest = xAll[121:221]
    yTest = yAll[121:221]

    train_errors = np.zeros((max_deg, 2))
    onlyClassifiers = []

    for deg in range(1, max_deg + 1):
        classifier, error = least_squares(xTrain, yTrain, deg)
        train_errors[deg - 1] = (deg, error)
        onlyClassifiers.append(classifier)

    best_classifier, validation_errors = validate(onlyClassifiers, xValidate, yValidate)

    red_label = mpatches.Patch(codlor='red', label='Train errors')
    blue_label = mpatches.Patch(color='blue', label='Validation errors')
    plt.plot(train_errors[:, 0], train_errors[:, 1], '-ro', validation_errors[:, 0], validation_errors[:, 1], '-bo')
    plt.legend(handles = [red_label,blue_label])
    plt.show()

    print(validation_errors)
    min_error = calc_error(onlyClassifiers[best_classifier], xTest, yTest)

    print(min_error)




