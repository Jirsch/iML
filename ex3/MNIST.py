import ex3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def prepareData():
    data = np.loadtxt('4vs7_data.txt', delimiter=' ')
    data[(data < 10)] = 0
    data[(data >= 10)] = 1
    labels = np.loadtxt('4vs7_labels.txt', delimiter=' ')
    labels = (labels + 1) / 2
    return data,labels

def findERM(data, labels):
    erm,erm_error = ex3.ERM(data, labels)
    (y_erm,x_erm) = divmod(erm,28)
    print('The classifier corresponding to the pixel (%d,%d) has %f percent error' % (x_erm,y_erm, erm_error))

def visualizeLosses(data, labels):
    losses = ex3.allZeroOneLosses(data,labels)         
    plt.imshow(losses.reshape(28, 28),cmap = matplotlib.cm.Greys_r)
    plt.colorbar()
    plt.show(block=True)

if __name__ == '__main__':
    data,labels = prepareData()
    findERM(data, labels)
    visualizeLosses(data, labels)
