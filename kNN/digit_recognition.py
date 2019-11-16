import numpy as np 
import os
from knn import knn

def img2vector(filename):
    vec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vec[0,32*i+j] = int(line[j])
    return vec

def readDataset(dir_path):
    files = os.listdir(dir_path)
    dataset_length = len(files)
    dataset = np.zeros((dataset_length,1024))
    labels = []
    for i,data_file in enumerate(files):
        labels.append(int(data_file[0]))
        dataset[i,:] = img2vector(os.path.join(dir_path,data_file))

    return dataset,np.array(labels)


if __name__ == "__main__":
    vec = img2vector('KNN/testDigits/0_0.txt')
    print(vec.shape)
    train_data,train_labels = readDataset('KNN/trainingDigits')
    test_data,test_labels = readDataset('KNN/testDigits')
    print('read done!')

    model = knn()
    predicts = model.predict(test_data,train_data,train_labels,5)

    acc = np.sum(predicts == test_labels) / len(predicts)
    print('accuracy is: %.2f' % acc)
