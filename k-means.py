import numpy as np
import PCA
import data_reader
import evaluation

class k_means:
    def __init__(self,data,label,centers=4):
        self.data=data
        self.label=label.astype(np.int32)
        self.center=np.zeros(centers,self.data.shape[1])

    def init_center(self):
        div=len(self.data)/len(self.center)
        for i in range(0,len(self.center)):
            self.center[i]=self.data[div*i]

    def distance(self,x,y):
        square=np.sum((x-y)*(x-y))
        return square


    def update(self):
        self.count=np.zeros(len(self.center))
        self.new_center=np.zeros(self.center.shape)
        for i in range(0,len(self.data)):
            min_index=0
            min_value=-1.0
            for j in range(0,len(self.center)):
                square=self.distance(self.center[j],self.data[i])
                if(square<min_value):
                    min_index=j
                    min_value=square
                self.new_center[min_index]+=self.data[i]
                self.count[min_index]+=1
        self.new_center=self.new_center/self.count
        self.center=self.new_center

    def predict_data(self):
        predict_index=np.zeros(len(self.data)).astype(np.int32)
        for i in range(0,len(self.data)):
            min_index=0
            min_value=-1.0
            for j in range(0,len(self.center)):
                square=self.distance(self.center[j],self.data[i])
                if(square<min_value):
                    min_index=j
                    min_value=square
                predict_index[i]=min_index

        new_predict=np.array(self.label.shape)
        for i in range(0,len(self.center)):
            self.current_label=self.label[predict_index==i]
            count = np.bincount(self.current_label)
            new_predict[predict_index==i]=np.argwhere(count=np.max(count))[0]
        return new_predict

if __name__ == '__main__':
    data,label_family,label_genus,label_species,label_record=data_reader.read_frog_data()
    predictor=k_means(data,label_family)
    predictor.init_center()
    iteration=100
    for i in range(0,iteration):
        predictor.update()
        perdict=predictor.predict_data()
        acc=evaluation.accuracy(predict,label_family)
        print("the acc in it &d is %.4f"%(i,acc))
