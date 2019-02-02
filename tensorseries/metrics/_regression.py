import numpy as np
import pandas as pd

__all__ = ['normal_loss', 'l1loss', 'l2loss']

def normal_loss(y_true, y_pred, k, log=False):
    if log:
        return np.power((np.log1p(y_true)-np.log1p(y_pred)).abs().pow(k).sum(), 1/k)
    else:
        return np.power((y_true-y_pred).abs().pow(k).sum(), 1/k)

def l1loss(y_true, y_pred, log=False):
    return normal_loss(y_true, y_pred, k=1, log=log)

def l2loss(y_true, y_pred):
    return normal_loss(y_true, y_pred, k=2, log=log)
