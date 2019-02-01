import numpy as np
import pandas as pd

__all__ = ['normal_loss', 'l1loss', 'l2loss']

def normal_loss(y_true, y_pred, k):
    return np.power((y_true-y_pred).abs().pow(k).sum(), 1/k)

def l1loss(y_true, y_pred):
    return normal_loss(y_true, y_pred, k=1)

def l2loss(y_true, y_pred):
    return normal_loss(y_true, y_pred, k=2)
