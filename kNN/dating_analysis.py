import numpy as np
from knn import knn

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    num_lines = len(lines)

    dataset = np.zeros((num_lines,3))
    labels = []
    index = 0
    for line in lines:
        list_line = line.split('\t')
        dataset[index,:] = list_line[0:3]
        labels.append(int(list_line[3]))
        index += 1

    return dataset,labels

def autonorm(dataset):
    minVal = dataset.min(0)
    maxVal = dataset.max(0)
    ranges = maxVal - minVal
    normDataset = (dataset - minVal) / ranges

    return normDataset,ranges,minVal 


if __name__ == "__main__":
    dataset,labels = file2matrix('kNN/datingTestSet2.txt')
    labels = np.array(labels)
    print(dataset.shape)
    print(labels.shape)

    dataset,ranges,minVal = autonorm(dataset)

    dataset_length = dataset.shape[0]
    orders = np.random.permutation(dataset_length)
    dataset = dataset[orders]
    labels = labels[orders]


    train_length = int(dataset_length * 0.95)
    train_data = dataset[0:train_length]
    test_data = dataset[train_length:]
    train_labels = labels[0:train_length]
    test_labels = labels[train_length:]

    model = knn()
    predicts = model.predict(test_data,train_data,train_labels,5)
    print(predicts)
    print(test_labels)

    acc = np.sum(predicts == test_labels) / len(predicts)
    print('accuracy is: %.2f' % acc)