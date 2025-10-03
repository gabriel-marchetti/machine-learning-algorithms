import numpy as np

def compute_f1(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    from .precision import compute_precision
    from .recall import compute_recall

    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)

    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)