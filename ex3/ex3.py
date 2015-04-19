import numpy as np


# This function computes the empirical 0-1 error of the classifier corresponding
# to the given coordinate.
#
# Input:
#   'coordinate': an int netween 0 to (n-1)
#   'data': a 0-1 martix of size n times m and of type numpy.ndarray.
#   Each column is an example
#   'labels': a 0-1 vector (of type numpy.ndarray) of size m. label[i] is the 
#   label of the i'th example
#
# Output:
#   'zero_one_loss': The empirical 0-1 error of the classifier corresponding
#   to 'coordinate'.
def zeroOneLoss(coordinate, data, labels):
    num_coordinates, num_samples, = data.shape
    num_errors = np.sum(data[coordinate, :] != labels)

    zero_one_loss = num_errors/num_samples
    return zero_one_loss


# This function computes the empirical 0-1 error of all the classifiers in our
# class.
#
# Input:
#   'data': A 0-1 martix of size n times m and of type numpy.ndarray.
#   Each column is an example
#   'labels': A 0-1 vector (of type numpy.ndarray) of size m. label[i] is the
#   label of the i'th example
#
# Output:
#   'zero_one_loss': A vector (of type numpy.ndarray) of size n such that
#   zero_one_loss[i] is the empirical 0-1 error of the classifier corresponding
#   to i.
def allZeroOneLosses(data, labels):
    num_coordinates, num_samples, = data.shape
    zero_one_loss = np.zeros(num_coordinates)

    for i in range(0, num_coordinates):
        zero_one_loss[i] = zeroOneLoss(i, data, labels)

    return zero_one_loss


# This function computes the ERM.
#
# Input:
#   'data': A 0-1 matrix of size n times m and of type numpy.ndarray.
#   Each column is an example
#   'labels': A 0-1 vector (of type numpy.ndarray) of size m. label[i] is the
#   label of the ith example
#
# Output:
#   'erm': The coordinate of the ERM classifier
#   'erm_error': The error of the ERM classifier
def ERM(data, labels):
    zero_one_loss = allZeroOneLosses(data, labels)

    erm = np.argmin(zero_one_loss)
    erm_error = zero_one_loss[erm]

    return erm, erm_error


# This function computes a bag-of-words representation of the given text.
#
# Input:
#   'text': A text file. Every line contains an SMS message and a label.
#   The first word in each line is either 'spam' or 'ham', indicating whether
#   the message is spam or not. The remaining part of line is the message.
#   'D': A python dictionary, mapping strings (words) to integers.
#   'num_samples': The number of examples. Equals to the number of
#   lines in 'text'
#
# Output:
#   'data': A 0-1 matrix of size len(D) times num_examples and of type
#   numpy.ndarray. Each column is an example. The ith column corresponds
#   to the ith message in 'text'. For every word w in D, if w appears in
#   the message, then the D[w]'th coordinate in that vector is 1. The rest of
#   the coordinates are 0.
#   'labels': A 0-1 vector (of type numpy.ndarray) of size m. label[i] is the
#   label of the ith message (1 if it is "spam" and 0 if it is "ham")
def vectorizeText(text, D, num_samples):
    vector_length = len(D)
    data = np.zeros((vector_length, num_samples))
    labels = np.zeros(num_samples)
    ln = 0
    for line in text:
        for word in line.split()[1:]:
            data[D[word], ln] = 1

        labels[ln] = 1 if (line.split(maxsplit=1)[0] == "spam") else 0
        ln += 1
    return data, labels
