import numpy as np 
import pandas as pd

from Temis.fairness_metrics.spd import compute_spd
from Temis.fairness_metrics.aod import compute_aod

def compare_fairness(models, X, y, sensitive_attr_index):
    """
    Compare the fairness of multiple models using Demographic Parity and Equalized Odds metrics.

    Parameters:
    models (dict): A dictionary where keys are model names and values are model instances with a predict method.
    X (pd.DataFrame): The input features including the sensitive attribute.
    sensitive_attr (str): The name of the sensitive attribute column in X.

    Returns:
    pd.DataFrame: A DataFrame containing the fairness metrics for each model.
    """
    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X)

        spd = compute_spd(y_pred, X[:, sensitive_attr_index])
        aod = compute_aod(y, y_pred, X[:, sensitive_attr_index])

        results.append({
            'Model': model_name,
            'Demographic Parity': spd,
            'Equalized Odds': aod
        })

    return pd.DataFrame(results)