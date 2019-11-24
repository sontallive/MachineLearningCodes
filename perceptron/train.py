import numpy as np

class Perceptron:

    def __init__(self,inX,labels):
        self.X = inX
        self.labels = labels
        self.w = np.zeros(inX.shape[1:])
        self.b = 0
        self.sign = lambda x:-1 if x < 0 else 1

    
    def train(self,max_iter_num=20,learning_rate=0.1):

        for _ in range(max_iter_num):
            for i in range(self.X.shape[0]):
                p = self.sign(np.dot(self.w,self.X[i]) + self.b)
                if p != self.labels[i]:
                    
                    self.w += learning_rate * self.labels[i] * self.X[i]
                    self.b += learning_rate * self.labels[i]
                    print('updated:w is {},b is {}'.format(self.w,self.b))

    def predict(self,x):
        return self.sign(np.dot(x, self.w) + self.b)


if __name__ == "__main__":
    datas = [
        [0,0],
        [1,0],
        [2,0],
        [1,1],
        [1,3],
        [3,1],
        [2,2]
    ]

    labels = [-1,-1,-1,-1,1,1,1]

    datas = np.array(datas)
    labels = np.array(labels)

    model = Perceptron(datas,labels)

    model.train()

    p = model.predict(np.array([1,5]))
    print(p)