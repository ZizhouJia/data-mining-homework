import numpy as np

def zero_mean(data_raw):
    mean=np.mean(data_raw,axis=0)
    data=data_raw-mean
    return data,mean

def pca(data_raw,n):
    data,mean=zeroMean(data_raw)
    cov=np.cov(data,rowvar=0)
    eig_vals,eig_vects=np.linalg.eig(np.mat(cov))
    eig_indice=np.argsort(eig_vals)
    n_indice=eig_indice[-1:-(n+1):-1]
    n_vects=eig_vects[:,n_indice]
    low_level_data=data*n_vect
    recont_data=(low_level_data*n_vect.T)+mean
    return low_level_data,recont_data
