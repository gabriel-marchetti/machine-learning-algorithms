import numpy as np

'''
Compute Disparate Impact Ratio (DIR) between privileged and unprivileged groups.
Parameters:
    y_pred (np.ndarray): Predicted binary labels (0 or 1).
    protected_attr (np.ndarray): Binary protected attribute (0 for unprivileged, 1 for privileged).
Returns:
    float: Disparate Impact Ratio (DIR).
'''
def compute_dir(y_pred : np.ndarray, protected_attr : np.ndarray) -> float:
    if not (y_pred.shape == protected_attr.shape):
        raise ValueError("Shapes of y_true, y_pred, and protected_attr must be the same.")
    
    p_priv = np.mean(y_pred[protected_attr == 1])
    p_unpriv = np.mean(y_pred[protected_attr == 0])
    
    if p_priv == 0:
        raise ValueError("No positive predictions for the privileged group, DIR is undefined.")

    return p_unpriv / p_priv