import data_reader
import numpy as np

def distance(x,y):
    x = np.array(x)
    y = np.array(y)
    square=np.sum((x-y)*(x-y))
    return square

def dbscan(data, eps, minpts):
    kernel_set = set()
    cluster_num = 0
    cluster = []
    unvisited_set = set(data)
    ii = 0
    for point in unvisited_set:
        print("initializatioin" + str(ii))
        if len([i for i in data if distance(point, i) <= eps]) >= minpts:
            kernel_set.add(point)
        ii = ii + 1

    while (len(kernel_set)):
        unvisited_set_pre = unvisited_set
        point = list(kernel_set)[np.random.randint(0, len(kernel_set))]
        unvisited_set = unvisited_set - set(point)
        queue = []
        queue.append(point)
        while len(queue):
            q = queue[0]
            q_adj = [i for i in data if distance(i, q) <= eps]
            if len(q_adj) > minpts:
                s = unvisited_set & set(q_adj)
                queue = queue + list(s)
                unvisited_set = unvisited_set - s
            queue.remove(q)
        cluster_num += 1
        cluster.append(list(unvisited_set_pre - unvisited_set))
        kernel_set = kernel_set - set(list(unvisited_set_pre - unvisited_set))
        print(cluster_num)
    return cluster

if __name__ == '__main__':
    data,label_family,label_genus,label_species,label_record=data_reader.read_frog_data()
    data_list = [tuple(data[i]) for i in range(0, data.shape[0])]
    print(len(data_list))
    cluster_list = dbscan(data=data_list, eps=0.1, minpts=500)
    print(len(cluster_list))
    for cluster in cluster_list:
        print(len(cluster))
