import numpy as np

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

