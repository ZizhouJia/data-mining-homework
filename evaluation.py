import numpy as np

def F_score(predict,label):
    return None

def accuracy(predict,label):
    sum=np.sum((predict==label).float())
    return sum/predict.shape[0]
