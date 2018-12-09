import numpy as np

def class_evaluation(prediction, ground_truth, threshold):
    mask = prediction>threshold
    prediction[mask] = 1
    prediction[~mask] = 0
    accuracy = float(np.sum(prediction==ground_truth))/float(ground_truth.shape[0])
    precision = float(np.sum((prediction==ground_truth)*prediction)) / float(np.sum(prediction))
    recall = float(np.sum((prediction==ground_truth)*prediction)) / float(np.sum(ground_truth))
    return accuracy, precision, recall