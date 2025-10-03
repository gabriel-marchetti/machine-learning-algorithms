import numpy as np 

def compute_precision(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)