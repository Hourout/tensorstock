import numpy as np
import pandas as pd

__all__ = ['accuracy', 'recall', 'precision', 'fbeta_score', 'f1_score',
           'auc_roc', 'auc_pr', 'sigmoid_crossentropy',
           'softmax_crossentropy']

def accuracy(y_true, y_pred):
    return (y_true==y_pred).mean()

def recall(y_true, y_pred, label=1):
    return y_pred[y_true==label].mean()

def precision(y_true, y_pred, label=1):
    return y_true[y_pred==label].mean()

def fbeta_score(y_true, y_pred, beta, label=1):
    r = recall(y_true, y_pred, label)
    p = precision(y_true, y_pred, label)
    return r*p*(1+np.power(beta, 2))/(np.power(beta, 2)*p+r)

def f1_score(y_true, y_pred, label=1):
    return fbeta_score(y_true, y_pred, beta=1, label=label)

def auc_roc(y_true, y_pred, label=1):
    assert y_true.nunique()==2, "`y_true` should be binary classification."
    t = pd.concat([y_true, y_pred], axis=1)
    t.columns = ['label', 'prob']
    t.insert(0, 'target', t[t.label!=label].label.unique()[0])
    t = t[t.label!=label].merge(t[t.label==label], on='target')
    auc = (t.prob_y>t.prob_x).mean()+(t.prob_y==t.prob_x).mean()/2
    return auc

def auc_pr(y_true, y_pred, label=1, prob=0.5):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob'], ascending=False).reset_index(drop=True)   
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['recall'] = t.tp/t.label.sum()
    t['precision'] = t.tp/(t.tp+t.fp)
    auc = t.sort_values(['recall', 'precision']).drop_duplicates(['recall'], 'last').precision.mean()
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
