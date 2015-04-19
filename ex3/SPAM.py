import ex3
import numpy as np

def prepareData():
    # Extract the words and prepare the dictionary
    f = open('SMSSpamCollection.txt', 'r')
    all_words = []
    text_length = 0
    for line in f:
        all_words.extend(line.split()[1:])
        text_length +=1
    all_words = sorted(set(all_words))
    D = dict(zip(all_words,range(len(all_words))))

    # Read the data
    data, labels = ex3.vectorizeText(open('SMSSpamCollection.txt', 'r'),D,text_length)
    return all_words, data, labels

def findTenBestClassifiers(all_words, data, labels):
    losses = ex3.allZeroOneLosses(data,labels)
    best_words = losses.argsort()[:10]
    #best_words = np.argpartition(losses, 10)[:10]
    #best_words = best_words[np.argsort(losses[best_words])]
    
    print ('The words that define the ten best classifiers are:')      
    print ('---------------------------------------------------')      
    for i in best_words:
        error = losses[i]
        word = all_words[i]
        print ('The classifier corresponding to "%s" has %f percent error' % (word, error))

if __name__ == '__main__':
    all_words,data,labels, = prepareData()
    findTenBestClassifiers(all_words, data, labels)