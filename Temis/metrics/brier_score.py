import numpy as np

def compute_brier(y_true : np.ndarray, y_prob : np.ndarray) -> float:
    """
    Compute the Brier score for binary classification.

    Parameters:
    y_true (np.ndarray): True binary labels (0 or 1).
    y_prob (np.ndarray): Predicted probabilities for the positive class (between 0 and 1).

    Returns:
    float: Brier score.
    """
    if y_true.shape != y_prob.shape:
        raise ValueError("Shapes of y_true and y_prob must be the same.")
    
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("All predicted probabilities must be between 0 and 1.")
    
    return np.mean((y_prob - y_true) ** 2)