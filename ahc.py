import numpy as np
from operator import itemgetter
from collections import OrderedDict,Counter
from PCA import pca
import data_reader
import time
import evaluation

def distance(x,y):
    #x = np.array(x)
    #y = np.array(y)
    square=np.sum((x-y)*(x-y))
    return square

def get_info(index, n):
    """
        i --> j
        return i, j 
    """
    return index/n, index%n

def info_encode(i, j, n):
    return i*n+j

# dsu
def get_root(fa, root):
    if (fa[root]==root):
        return root
    else:
        fa[root]=get_root(fa, fa[root])
        return fa[root]

def agnes(data, clusters):
    # initialization

    # init dsu:
    n = len(data)
    fa = [i for i in range(n)]

    sorted_distance_dic = []

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            sorted_distance_dic.append([distance(data[i], data[j]), info_encode(i, j, n)])

    # sort distance
    sorted_distance_dic.sort()
    print("finish sorting")

    prediction = [i for i in range(data.shape[0])]

    cluster_num = len(data)

    for dis, info in sorted_distance_dic:
        if cluster_num <= clusters:
            break
        x, y = get_info(info, n)
        X = get_root(fa, x)
        Y = get_root(fa, y)
        if X != Y:
            fa[X] = Y
            cluster_num = cluster_num -1
    for i in range(n):
        get_root(fa, i)
    return fa

if __name__ == '__main__':
    clusters=700
    data,label_family,label_genus,label_species,label_record=data_reader.read_frog_data()
    data=pca(data,10)
    data=data/data.max(axis=0)
    res = [0 for _ in range(data.shape[0])]
    start_time = time.time()
    cluster_result = agnes(data, clusters)

    cluster_set = set(cluster_result)

    ii = 0
    for index in cluster_set:
        cluster_indexs = [i for i, x in enumerate(cluster_result) if x == index]
        for cluster_index in cluster_indexs:
            res[cluster_index] = ii
        ii = ii + 1
    res = np.array(res)
    predict = res
    train_time = time.time() - start_time
    label=np.argmin(-label_family,axis=1)
    acc=evaluation.multi_label_accuracy(predict,label,clusters)
    p=evaluation.purity(predict,label,clusters)
    F_score=evaluation.F_score(predict,label,clusters)
    F_score_output=evaluation.format_F_score(F_score)
    print("The acc is %.4f, purity is %.4f"%(acc,p))
    print("The F-score is " + F_score_output)

    print("Total time:")
    print(train_time)