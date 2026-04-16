import numpy as np 

def calders_verwer_score(y_pred, s_attr):
    pr_s1 = y_pred[s_attr == 1].mean()
    pr_s0 = y_pred[s_attr == 0].mean()

    return pr_s1 - pr_s0