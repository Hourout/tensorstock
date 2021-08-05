__all__ = ['SMA', 'EMA', 'WMA', 'DEMA']

def SMA(series, window):
    return series.rolling(window=window).mean()

def EMA(series, window):
    return series.ewm(span=window).mean()

def WMA(series, window):
    return series.rolling(window=window).apply(lambda x:(x*x/x.sum()).sum())

def DEMA(series, window):
    return 2*EMA(series, window)-EMA(EMA(series, window), window)