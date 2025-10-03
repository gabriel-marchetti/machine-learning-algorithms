import numpy as np

def compute_recall(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)