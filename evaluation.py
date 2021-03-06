import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def class_evaluation(prediction, ground_truth, threshold):
    prediction=prediction.reshape(-1)
    ground_truth=ground_truth.reshape(-1)
    mask = prediction>threshold
    prediction[mask] = 1
    prediction[~mask] = 0
    accuracy = float(np.sum(prediction==ground_truth))/float(ground_truth.shape[0])
    precision =0
    if(float(np.sum(prediction))!=0):
         precision=float(np.sum((prediction==ground_truth)*prediction)) / float(np.sum(prediction))
    recall = float(np.sum((prediction==ground_truth)*prediction)) / float(np.sum(ground_truth))
    return accuracy, precision, recall

def multi_label_accuracy(prediction,ground_truth,clusters):
    predict_index=prediction
    label=ground_truth
    new_predict=np.zeros(label.shape).astype(np.int32)
    for i in range(0,clusters):
        current_label=label[predict_index==i]
        count = np.bincount(current_label)
        count=list(count)
        new_predict[predict_index==i]=count.index(max(count))
    sum=np.sum((new_predict==label).astype(np.float32))
    return sum/new_predict.shape[0]

def purity(prediction,ground_truth,clusters):
    predict_index=prediction
    label=ground_truth
    sum=0.0
    for i in range(clusters):
        current_label=label[predict_index==i]
        count = np.bincount(current_label)
        count=list(count)
        max_class=count.index(max(count))
        sum+=np.sum((current_label==max_class).astype(np.float32))/current_label.shape[0]
    sum=sum/clusters
    return sum

def F_score(prediction,ground_truth,clusters,classes=4):
    predict_index=prediction
    label=ground_truth
    new_predict=np.zeros(label.shape).astype(np.int32)
    for i in range(0,clusters):
        current_label=label[predict_index==i]
        count = np.bincount(current_label)
        count=list(count)
        new_predict[predict_index==i]=count.index(max(count))
    predict=new_predict
    F_score=[]
    for i in range(0,classes):
        predict_real_label=label[predict==i]
        if(predict_real_label.shape[0]==0):
            F_score.append(0)
            continue
        precision=np.sum((predict_real_label==i).astype(np.float32))/predict_real_label.shape[0]
        real_label_predict=predict[label==i]
        recall=np.sum((real_label_predict==i).astype(np.float32))/real_label_predict.shape[0]
        f=2*precision*recall/(precision+recall)
        F_score.append(f)
    return F_score

def format_F_score(F_score):
    string=""
    for score in F_score:
        string+=str(score)+","
    return string[:-1]

def ROC(prediction,label):
    prediction=prediction.reshape(-1)
    label=label.reshape(-1).astype(np.int)
    fpr,tpr,thre = roc_curve(label,prediction)
    auc_value = auc(fpr,tpr)
    plt.plot(fpr,tpr,color = 'darkred',label = 'roc area:(%0.2f)'%auc_value)
    plt.plot([0,1],[0,1],linestyle = '--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('roc_curve')
    plt.legend(loc = 'lower right')
    plt.show()
