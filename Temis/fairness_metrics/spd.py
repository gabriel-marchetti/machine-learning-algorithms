import numpy as np

'''
Compute Statistical Parity Difference (SPD) between privileged and unprivileged groups.
Parameters:
    y_pred (np.ndarray): Predicted binary labels (0 or 1).
    protected_attr (np.ndarray): Binary protected attribute (0 for unprivileged, 1 for privileged).
Returns:
    float: Statistical Parity Difference (SPD).
'''
def compute_spd(y_pred : np.ndarray, protected_attr : np.ndarray) -> float:
    if not (y_pred.shape == protected_attr.shape):
        raise ValueError("Shapes of y_true, y_pred, and protected_attr must be the same.")
    
    p_priv = np.mean(y_pred[protected_attr == 1])
    p_unpriv = np.mean(y_pred[protected_attr == 0])
    
    return p_unpriv - p_priv