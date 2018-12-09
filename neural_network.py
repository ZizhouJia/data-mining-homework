from data_reader import *
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from evaluation import class_evaluation

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Net(torch.nn.Module):
    def __init__(self, n_feature=61, n_hidden=256, n_output=1):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden//2)
        self.out = torch.nn.Linear(n_hidden//2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    data, label = read_bank_data()
    data_pro, label_pro = [], []
    data_size = data.shape[0]

    #for i in range(data_size):
    #    if label[i]==0:
    #        data_pro.append(data[i])
    #        label_pro.append(label[i])
    #    else:
    #        for k in range(10):
    #            data_pro.append(data[i])
    #            label_pro.append(label[i])

    for i in range(data_size):
        if label[i]==0:
            if i % 10 == 0:
                data_pro.append(data[i])
                label_pro.append(label[i])
        else:
                data_pro.append(data[i])
                label_pro.append(label[i])

    data_pro = np.asarray(data_pro, dtype=np.float32)
    label_pro = np.asarray(label_pro, dtype=np.float32)
    data_pro = data_pro.reshape((-1, 61))
    label_pro = label_pro.reshape((-1,1))
    data_pro_size = data_pro.shape[0]
    print(data_pro_size)
    data_pro = data_pro/data_pro.max(axis=0)

    test_pro_size = int(0.3 * data_pro_size)
    train_pro_size = data_pro_size - test_pro_size
    test_index = np.asarray(random.sample(range(data_pro_size), test_pro_size))
    train_index = np.asarray(list(set(range(data_pro_size))^set(test_index)))

    x_train, y_train = torch.from_numpy(data_pro[train_index]).float(), torch.from_numpy(label_pro[train_index]).float()
    x_test, y_test = torch.from_numpy(data_pro[test_index]).float(), torch.from_numpy(label_pro[test_index]).float()
    print(y_train)
    net = Net()     # define the network
    print(net)  # net architecture
    net.apply(init_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.03)
    loss_func = torch.nn.BCELoss()

    #data_pro = np.random.rand(data_pro_size, 61)
    batch_size = 200
    for epoch in range(20):
        x_train_index = random.sample(train_index, train_pro_size)
        x_train = torch.from_numpy(data_pro[x_train_index])
        y_train = torch.from_numpy(label_pro[x_train_index])
        for j in range(train_pro_size // batch_size):
            x = Variable(x_train[batch_size * j:batch_size * (j+1)])
            y = Variable(y_train[batch_size * j:batch_size * (j+1)])
            out = net(x)
            loss = loss_func(out, y)
            #if j==0:
                #print("-----loss-----")
                #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    prediction = net(x_test)
    #mask = prediction>0.5
    #prediction[mask] = 1
    #prediction[~mask] = 0
    #print(prediction)
    #print(y_test.shape[0])
    #print(torch.sum(prediction==y_test))

    accuracy, precision, recall = class_evaluation(prediction=prediction.detach().numpy(),
                                                   ground_truth=y_test.detach().numpy(),
                                                   threshold=0.5)
    print(accuracy)
    print(precision)
    print(recall)
