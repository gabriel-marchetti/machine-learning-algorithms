import numpy as np

'''
Compute Absolute Average Odds Difference (AAOD) between privileged and unprivileged groups.
Parameters:
    y_true (np.ndarray): True binary labels (0 or 1).
    y_pred (np.ndarray): Predicted binary labels (0 or 1).
    protected_attr (np.ndarray): Binary protected attribute (0 for unprivileged, 1 for privileged).
Returns:
    float: Absolute Average Odds Difference (AAOD).
'''
def compute_aaod(y_true : np.ndarray, y_pred : np.ndarray, protected_attr : np.ndarray) -> float:
    if not (y_true.shape == y_pred.shape == protected_attr.shape):
        raise ValueError("Shapes of y_true, y_pred, and protected_attr must be the same.")
    
    # Stands for true positive privileged
    tp_priv = np.sum((y_pred == 1) & (y_true == 1) & (protected_attr == 1))
    # Stands for false positive privileged
    fp_priv = np.sum((y_pred == 1) & (y_true == 0) & (protected_attr == 1))

    # Stands for true positive unprivileged
    tp_unpr = np.sum((y_pred == 1) & (y_true == 1) & (protected_attr == 0))
    # Stands for false positive unprivileged
    fp_unpr = np.sum((y_pred == 1) & (y_true == 0) & (protected_attr == 0))

    # Stands for true privileged, i.e., Y = 1 and protected = 1
    t_priv = np.sum((protected_attr == 1) & (y_true == 1))
    # Stands for true unprivileged, i.e., Y = 1 and protected = 0
    t_unpr = np.sum((protected_attr == 0) & (y_true == 1))
    # Stands for false privileged, i.e., Y = 0 and protected = 1
    f_priv = np.sum((protected_attr == 1) & (y_true == 0))
    # Stands for false unprivileged, i.e., Y = 0 and protected = 0
    f_unpr = np.sum((protected_attr == 0) & (y_true == 0))

    if t_priv == 0 or t_unpr == 0 or f_priv == 0 or f_unpr == 0:
        raise ValueError("No positive/negative samples for one of the groups, AOD is undefined.")
    p_tp_priv = tp_priv / t_priv 
    p_tp_unpr = tp_unpr / t_unpr
    p_fp_priv = fp_priv / f_priv
    p_fp_unpr = fp_unpr / f_unpr

    
    # Average Odds Difference
    aod = 1/2 * (abs(p_tp_unpr - p_tp_priv) + abs(p_fp_unpr - p_fp_priv))
    
    return aod