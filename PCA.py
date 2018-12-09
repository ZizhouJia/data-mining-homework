import numpy as np

def zero_mean(data_raw):
    mean=np.mean(data_raw,axis=0)
    data=data_raw-mean
    return data,mean

def pca(data_raw,n):
    data,mean=zero_mean(data_raw)
    cov=np.cov(data,rowvar=0)
    eig_vals,eig_vects=np.linalg.eig(np.mat(cov))
    eig_indice=np.argsort(-eig_vals)
    n_indice=eig_indice[0:n]
    n_vects=eig_vects[:,n_indice]
    low_level_data=data*n_vects
    low_level_data=np.array(low_level_data)
    return low_level_data
