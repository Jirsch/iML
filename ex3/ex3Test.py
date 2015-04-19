import ex3
import numpy as np

def prepareData():
    data = np.loadtxt('4vs7_data.txt', delimiter=' ')
    data[(data < 10)] = 0
    data[(data >= 10)] = 1
    labels = np.loadtxt('4vs7_labels.txt', delimiter=' ')
    labels = (labels + 1) / 2
    return data,labels

if __name__ == '__main__':
    data,labels = prepareData()
    erm,erm_error = ex3.ERM(data, labels)
    if erm_error <= 0.12:
        print("PASS: Your code passed the test.")
    else:
        print("FAIL: The loss of the returned hypothesis is higher than what is expeted")
