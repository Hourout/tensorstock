import numpy as np
import pandas as pd

__all__ = ['accuracy', 'recall', 'precision', 'auc', 'sigmoid_crossentropy',
           'softmax_crossentropy']

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

def sigmoid_crossentropy(y_true, y_pred):
    t = np.exp(y_pred-np.max(y_pred))
    t = -(np.log(t/t.sum())*y_true).mean()
    return t

def softmax_crossentropy(y_true, y_pred, one_hot=True):
    assert y_pred.shape[1]==y_true.nunique(), "`y_pred` and `y_true` dim not same."
    t = np.exp(y_pred.T-np.max(y_pred, axis=1))
    if one_hot:
        t = -(np.log(t/np.sum(t, axis=0)).T*pd.get_dummies(y_true)).sum(axis=1).mean()
    else:
        t = -(np.log(t/np.sum(t, axis=0)).T*y_true).sum(axis=1).mean()
    return t
