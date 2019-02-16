import pandas as pd

__all__ = ['equal_width', 'equal_frequency']

def equal_width(x, bins, one_hot=False, one_hot_name='columns'):
    t = pd.qcut(x, bins, labels=range(bins))
    if one_hot:
        t = pd.get_dummies(t, prefix=one_hot_name)
    return t

def equal_frequency(x, bins, one_hot=False, one_hot_name='columns'):
    t = pd.cut(x, bins, labels=range(bins))
    if one_hot:
        t = pd.get_dummies(t, prefix=one_hot_name)
    return t
