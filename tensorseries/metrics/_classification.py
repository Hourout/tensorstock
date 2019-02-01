import numpy as np
import pandas as pd

__all__ = ['accuracy', 'recall', 'precision']

def accuracy(y_true, y_pred):
    return (x_true==y_pred).mean()

def recall(y_true, y_pred, label=1):
    return np.mean(y_true[y_true==label].index==y_pred[y_true==label].index)

def precision(y_true, y_pred, label=1):
    return np.mean(y_true[y_pred==label].index==y_pred[y_pred==label].index)
    
    
