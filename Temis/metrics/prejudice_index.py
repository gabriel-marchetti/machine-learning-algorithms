import numpy as np
from sklearn.metrics import mutual_info_score

def entropy(labels):
    pr = np.bincount(labels) / len(labels)
    pr = pr[pr > 0]
    return -np.sum(pr * np.log2(pr))

def normalized_prejudice_index(y_pred, s_attr):
    pi = mutual_info_score(s_attr, y_pred)

    h_y = entropy(y_pred)
    h_s = entropy(s_attr)
    return pi / np.sqrt(h_y * h_s) if h_y > 0 and h_s > 0 else 0.0