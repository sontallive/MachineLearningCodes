import numpy as np
import time

class KMeans:
    def __init__(self,data,k,iter_num = 50):
        self.data = data
        self.length = data.shape[0]
        self.k = k
        choice = np.random.choice(a=self.length, size=self.k, replace=False)
        self.centroids = self.data[choice]
        self.labels = np.zeros(self.length)
        self.MAX_ITER_NUM = iter_num

    def compute_centroids(self):
        for i in range(self.k):
            ids = (self.labels == i)
            x = self.data[ids]
            self.centroids[i,:] = np.mean(x,0)
            

    def update_label(self):
        # print('start to update label...',end=' ')
        tick = time.time()
        for i in range(self.length):
            # print('\rstart to update label %d/%d' % (i,self.length),end=" ")
            dist = np.sum(np.abs(self.data[i,:] - self.centroids),axis = 1)
            # print(dist.shape) 
            self.labels[i] = np.argmin(dist)
        print('time used:%ds' % int(time.time() - tick))
        

    def run(self):
        
        for i in range(self.MAX_ITER_NUM):
            last_centroids = self.centroids.copy()
            print('K-Means iteration %d/%d..' % (i+1,self.MAX_ITER_NUM))
            self.update_label()
            self.compute_centroids()
            move_step = np.mean(np.abs(last_centroids-self.centroids))
            # print(model.labels)W
            if move_step < 0.01 :
                print("didn't change... leave iteration...")
                break
            print('move step:',move_step)
        

if __name__ == "__main__":
    data = np.random.randn(100,128)
    print(data.shape)
    model = KMeans(data,20,iter_num=50)
    model.run()
    print(model.labels)
    
