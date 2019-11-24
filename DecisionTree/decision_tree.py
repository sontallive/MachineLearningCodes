from math import log


def calcShannonEnt(dataset):
    data_length = len(dataset)
    label_counts = {}

    for featVec in dataset:
        label = featVec[-1]
        if label not in label_counts.keys():
            label_counts[label] = 0
        label_counts[label] += 1
    shannon_ent = 0.0
    for label in label_counts:
        prob = label_counts[label] / data_length
        shannon_ent -= prob * log(prob,2)

    return shannon_ent

def splitData(dataset,axis,value):
    retDataset = []
    for vec in dataset:
        if vec[axis] == value:
            reduced_vec = vec[:axis]
            reduced_vec.extend(vec[axis+1:])
            retDataset.append(reduced_vec)
    return retDataset

def chooseBestFeatureToSplit(dataset):
    num_features = len(dataset[0]) - 1
    best_info_gain = 0.0
    base_ent = calcShannonEnt(dataset)
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_ent = 0.0
        for val in unique_vals:
            sub_dataset = splitData(dataset,i,val)
            prob = len(sub_dataset) / len(dataset)
            new_ent += prob * calcShannonEnt(sub_dataset)
        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    
    return best_feature

def createDataset():
    dataset = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['no surfacing','flippers']

    return dataset,labels



if __name__ == "__main__":
    dataset,labels = createDataset()
    print(calcShannonEnt(dataset))

    print(splitData(dataset,2,'no'))

    axis = chooseBestFeatureToSplit(dataset)
    print(axis)