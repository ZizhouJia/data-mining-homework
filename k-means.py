import numpy as np
import PCA
import data_reader
import evaluation

class k_means:
    def __init__(self,data,label,centers=40):
        self.data=data
        self.label=label
        self.center=np.zeros((centers,self.data.shape[1]))

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
            min_value=1e100
            for j in range(0,len(self.center)):
                square=self.distance(self.center[j],self.data[i])
                if(square<min_value):
                    min_index=j
                    min_value=square
            self.new_center[min_index]+=self.data[i]
            self.count[min_index]+=1
        self.new_center=self.new_center/self.count.reshape((-1,1))
        self.center=self.new_center

    def predict_data(self):
        predict_index=np.zeros(len(self.data)).astype(np.int32)
        for i in range(0,len(self.data)):
            min_index=0
            min_value=1e100
            for j in range(0,len(self.center)):
                square=self.distance(self.center[j],self.data[i])
                if(square<min_value):
                    min_index=j
                    min_value=square
            predict_index[i]=min_index

        return predict_index

if __name__ == '__main__':
    clusters=100
    data,label_family,label_genus,label_species,label_record=data_reader.read_frog_data()
    label=np.argmin(-label_family,axis=1)
    predictor=k_means(data,label,clusters)
    predictor.init_center()
    iteration=100
    for i in range(0,iteration):
        predictor.update()
        predict=predictor.predict_data()
        acc=evaluation.multi_label_accuracy(predict,label,clusters)
        p=evaluation.purity(predict,label,clusters)
        F_score=evaluation.F_score(predict,label,clusters)
        F_score_output=evaluation.format_F_score(F_score)
        print("the acc in it %d is %.4f %.4f"%(i,acc,p))
        print("the Fscore is "+F_score_output)
