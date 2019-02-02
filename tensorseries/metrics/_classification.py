import numpy as np
import pandas as pd

__all__ = ['accuracy', 'recall', 'precision', 'auc']

def accuracy(y_true, y_pred):
    return (x_true==y_pred).mean()

def recall(y_true, y_pred, label=1):
    return np.mean(y_true[y_true==label].index==y_pred[y_true==label].index)

def precision(y_true, y_pred, label=1):
    return np.mean(y_true[y_pred==label].index==y_pred[y_pred==label].index)
    
def auc(y_true, y_pred, label=1):
    assert y_true.nunique()==2, "`y_true` should be binary classification."
    t = pd.concat([y_true, y_pred], axis=1)
    t.columns = ['label', 'prob']
    t.insert(0, 'target', t[t.label!=label].label.unique()[0])
    t = t[t.label!=label].merge(t[t.label==label], on='target')
    auc = (t.prob_y>t.prob_x).mean()+(t.prob_y==t.prob_x).mean()/2
    return auc    
