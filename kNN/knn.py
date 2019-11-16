import numpy as np 


class knn:
    def predict(self,inX,dataset,labels,K):
        # dataset_length = dataset.shape[0]
        X_length = inX.shape[0]
        predicts = np.zeros(X_length)
        for i in range(X_length):
            x = inX[i,:]
            distances = np.sum((x - dataset)**2,axis=1)
            index_order = np.argsort(distances)[0:K]
            predicts[i] = np.argmax(np.bincount(labels[index_order]))
            
        return predicts

if __name__ == "__main__":

    dataset = np.array([[0.1,0.2],[0.3,1],[1.3,0.9],[1.1,1.0]])
    labels = np.array([0,0,1,1])

    X = np.array([[0.3,0.2],[1.4,1.3],[-0.2,0.5]])

    model = knn()

    predicts = model.predict(X,dataset,labels,2)
    print(predicts)

