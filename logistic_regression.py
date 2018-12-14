import numpy as np
import data_reader
import evaluation

class bank_dataset:
    def __init__(self):
        self.data,self.label=data_reader.read_bank_data()
        # data_part1=data_reader.process_number_to_label(self.data[:,0:9])
        # data_part2=self.data[:,9:]
        # self.data=np.concatenate((data_part1,data_part2),axis=1)
        self.label=np.reshape(self.label,(-1))
        self.data_neg=self.data[self.label<0.5]
        self.data_pos=self.data[self.label>0.5]
        self.data_pos=np.repeat(self.data_pos,int(len(self.data_neg)/len(self.data_pos)),axis=0)
        self.label_pos=np.ones(len(self.data_pos))
        self.label_neg=np.zeros(len(self.data_neg))
        self.data=np.concatenate((self.data_pos,self.data_neg),axis=0)
        self.label=np.concatenate((self.label_pos,self.label_neg),axis=0)
        state=np.random.get_state()
        np.random.shuffle(self.data)
        np.random.set_state(state)
        np.random.shuffle(self.label)
        self.data=self.normalize(self.data)
        self.train_X=[]
        self.train_Y=[]
        self.test_X=[]
        self.test_Y=[]
        for i in range(0,len(self.data)):
            if(i%10>=0 and i%10<=6):
                self.train_X.append(self.data[i])
                self.train_Y.append(self.label[i])
            else:
                self.test_X.append(self.data[i])
                self.test_Y.append(self.label[i])
        self.train_X=np.array(self.train_X)
        self.test_X=np.array(self.test_X)
        self.train_Y=np.array(self.train_Y)
        self.test_Y=np.array(self.test_Y)
        self.index=0

    def next_batch(self,batch_size=32):
        if(self.index+batch_size<len(self.train_X)):
            begin=self.index
            self.index+=batch_size
            return self.train_X[begin:self.index],self.train_Y[begin:self.index],False
        else:
            self.index=batch_size
            return self.train_X[0:self.index],self.train_Y[0:self.index],True

    def get_test_set(self):
        return self.test_X,self.test_Y

    def get_input_dim(self):
        return self.train_X.shape[1]

    def normalize(self,v):
        e=1e-8
        # mean=np.mean(v[:,0:9],axis=0)
        # std=np.std(v[:,0:9],axis=0,ddof=1)
        # v[:,0:9]=(v[:,0:9]-mean)/(std+e)
        mean=np.mean(v,axis=0)
        std=np.std(v,axis=0,ddof=1)
        v=(v-mean)/(std+e)
        return v


class logistic_regression:
    def __init__(self,input_dim):
        self.W=np.random.randn(input_dim,1)/100
        self.b=0.0

    def predict(self,input):
        output=np.dot(input,self.W)+self.b
        output=1.0/(1.0+np.exp(-output))
        return output

    def train(self,input,label,learning_rate=0.1,l2=0.1):
        label=np.reshape(label,(-1,1))
        batch_size=input.shape[0]
        output=self.predict(input)
        cha=label-output
        self.W=self.W*(1-learning_rate*l2/batch_size)+learning_rate*np.sum(cha*input,axis=0).reshape(-1,1)/batch_size
        self.b=self.b*(1-learning_rate*l2/batch_size)+learning_rate*np.sum(cha)/batch_size

    def calculate_acc(self,input,label):
        label=np.reshape(label,(-1,1))
        output=self.predict(input)
        output[output>0.5]=1.0
        output[output<0.5]=0.0
        sum=np.sum((output==label).astype(np.float32))
        return sum/label.shape[0]

    def calculate_loss(self,input,label):
        label=np.reshape(label,(-1,1))
        output=self.predict(input)
        e=1e-8
        loss=-label*np.log(output+e)-(1-label)*np.log(1-output+e)
        loss2=np.mean(self.W*self.W)
        return np.mean(loss),loss2




if __name__ == '__main__':
    dataset=bank_dataset()
    classifier=logistic_regression(dataset.get_input_dim())
    epoch=100
    iteration=1
    for i in range(0,epoch):
        epoch_end_mark=False
        while(not epoch_end_mark):
            data,label,epoch_end_mark=dataset.next_batch()
            classifier.train(data,label)
        test_data,test_label=dataset.get_test_set()
        prediction=classifier.predict(test_data)
        acc,precision,recall=evaluation.class_evaluation(prediction,test_label,0.5)
        loss,loss2=classifier.calculate_loss(test_data,test_label)
        print("in epoch %d the acc is: %.4f %.4f %.4f the loss is: %.4f l2 loss is: %.4f"%(i,acc,precision,recall,loss,loss2))
    test_data,test_label=dataset.get_test_set()
    prediction=classifier.predict(test_data)
    evaluation.ROC(prediction,test_label)
