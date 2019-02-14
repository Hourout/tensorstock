import numpy as np
import pandas as pd
import pyecharts as pe

__all__ = ['ks_curve']

def ks_curve(y_true, y_pred, label=1, prob=0.5, html=False, html_path='Kolmogorov-Smirnov Curve.html'):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob'], ascending=False).reset_index(drop=True)   
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['tpr'] = t.tp/t.label.sum()
    t['fpr'] = t.fp/(t.label.count()-t.label.sum())
    t.index = np.round(((t.index+1)/len(t))*100).astype(str)+"%"
    line = pe.Line("Kolmogorov-Smirnov Curve")
    line.add("TPR", t.index, t.tpr.round(4), is_smooth=True)
    line.add("FPR", t.index, t.fpr.round(4), is_smooth=True)
    line.add("KS", t.index, (t.tpr-t.fpr).abs().round(4), is_smooth=True, mark_point=["max"])
    return line if not html else line.render(html_path)
