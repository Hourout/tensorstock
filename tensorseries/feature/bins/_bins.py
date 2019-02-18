import numpy as np
import pandas as pd
from scipy.stats import chi2

__all__ = ['equal_width', 'equal_frequency', 'cut_box', 'chi_square_interval', 'chi_square_threshold']

def equal_width(feature, bins, one_hot=False, one_hot_name='columns'):
    t, bins = pd.qcut(feature, bins, labels=range(bins), retbins=True)
    if one_hot:
        t = pd.get_dummies(t, prefix=one_hot_name)
    return t, bins

def equal_frequency(feature, bins, one_hot=False, one_hot_name='columns'):
    t, bins = pd.cut(feature, bins, labels=range(bins), retbins=True)
    if one_hot:
        t = pd.get_dummies(t, prefix=one_hot_name)
    return t, bins

def cut_box(feature, bins, one_hot=False, one_hot_name='columns'):
    t = pd.Series(pd.cut(feature, bins, labels=range(len(bins)-1)).get_values()).fillna(-1)
    if one_hot:
        t = pd.get_dummies(t, prefix=one_hot_name)
    return t

def merge_chiSquare(df, merge_idx, idx):
    df.loc[idx, 'actual_0'] = df.at[idx, 'actual_0'] + df.at[merge_idx, 'actual_0']
    df.loc[idx, 'actual_1'] = df.at[idx, 'actual_1'] + df.at[merge_idx, 'actual_1']
    df.loc[idx, 'expect_0'] = df.at[idx, 'expect_0'] + df.at[merge_idx, 'expect_0']
    df.loc[idx, 'expect_1'] = df.at[idx, 'expect_1'] + df.at[merge_idx, 'expect_1']
    chi2_0 = (df.at[idx, 'expect_0'] - df.at[idx, 'actual_0'])**2 / df.at[idx, 'expect_0']
    chi2_1 = (df.at[idx, 'expect_1'] - df.at[idx, 'actual_1'])**2 / df.at[idx, 'expect_1']
    df.at[idx, 'chi'] = chi2_0 + chi2_1
    df = df.drop([merge_idx]).reset_index(drop=True)
    return df

def chi_square_interval(feature, label, max_interval=5):
    t = pd.DataFrame({'label':label, 'feature':feature})
    rate = label.sum()/label.count()
    t = t.groupby(['feature']).label.agg(['count', 'sum']).reset_index()
    t.columns = ['feature', 'actual_all', 'actual_1']
    t['actual_0'] = t.actual_all-t.actual_1
    t['expect_1'] = t.actual_all*rate
    t['expect_0'] = t.actual_all*(1-rate)
    t['chi'] = np.square(t.expect_0 - t.actual_0)/t.expect_0+np.square(t.expect_1 - t.actual_1)/t.expect_1
    while(len(t) > max_interval):
        min_index = t[t.chi==t.chi.min()].index.tolist()[0]
        if min_index == 0:
            t = merge_chiSquare(t, min_index+1, min_index)
        elif min_index == len(t)-1:
            t = merge_chiSquare(t, min_index-1, min_index)
        else:
            if t.loc[min_index-1, 'chi'] >= t.loc[min_index+1, 'chi']:
                t = merge_chiSquare(t, min_index, min_index+1)
            else:
                t = merge_chiSquare(t, min_index-1, min_index)
    t.loc[t.feature==t.feature.min(), 'feature'] = t.feature.min()-0.0001
    t, bins = pd.cut(feature, t.feature.values, labels=range(t.feature.count()-1), retbins=True)
    return t, bins

def chi_square_threshold(feature, label, dfree=4, cf=0.1, min_interval=4):
    t = pd.DataFrame({'label':label, 'feature':feature})
    rate = label.sum()/label.count()
    t = t.groupby(['feature']).label.agg(['count', 'sum']).reset_index()
    t.columns = ['feature', 'actual_all', 'actual_1']
    t['actual_0'] = t.actual_all-t.actual_1
    t['expect_1'] = t.actual_all*rate
    t['expect_0'] = t.actual_all*(1-rate)
    t['chi'] = np.square(t.expect_0 - t.actual_0)/t.expect_0+np.square(t.expect_1 - t.actual_1)/t.expect_1
    threshold = chi2.isf(cf, dfree)
    while(t.chi.min() < threshold and len(t) > min_interval):
        min_index = t[t.chi==t.chi.min()].index.tolist()[0]
        if min_index == 0:
            t = merge_chiSquare(t, min_index+1, min_index)
        elif min_index == len(t)-1:
            t = merge_chiSquare(t, min_index-1, min_index)
        else:
            if t.at[min_index-1, 'chi'] >= t.at[min_index+1, 'chi']:
                t = merge_chiSquare(t, min_index, min_index+1)
            else:
                t = merge_chiSquare(t, min_index-1, min_index)
    t.loc[t.feature==t.feature.min(), 'feature'] = t.feature.min()-0.0001
    t, bins = pd.cut(feature, t.feature.values, labels=range(t.feature.count()-1), retbins=True)
    return t, bins
