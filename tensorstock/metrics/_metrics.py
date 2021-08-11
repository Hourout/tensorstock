__all__ = ['accuracy_interval']

def accuracy_interval(true_highest, true_lowest, pred_highest, pred_lowest, 
                      value=[-3.0, -1.0, 0, 1.0, 4.0], weight=None):
    if weight is None:
        weight = [0.5+value[0]/20]+[(j-value[i])/20 for i,j in enumerate(value[1:])]+[0.5-value[-1]/20]
    true_high = ([1 if true_highest<i else 0 for i in value]+[1 if true_highest>=value[-1] else 0]).index(1)
    true_low = ([1 if true_lowest<i else 0 for i in value]+[1 if true_lowest>=value[-1] else 0]).index(1)
    pred_high = ([1 if pred_highest<i else 0 for i in value]+[1 if pred_highest>=value[-1] else 0]).index(1)
    pred_low = ([1 if pred_lowest<i else 0 for i in value]+[1 if pred_lowest>=value[-1] else 0]).index(1)
    label_true = [1 if i>=true_low and i <= true_high else 0 for i in range(6)]
    label_pred = [1 if i>=pred_low and i <= pred_high else 0 for i in range(6)]
    prob = round(sum([k for i,j,k in zip(label_true, label_pred, weight) if i==j]),3)
    return prob