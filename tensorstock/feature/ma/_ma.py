__all__ = ['SMA', 'EMA', 'WMA', 'DEMA', 'DMA']

def SMA(series, window):
    return series.rolling(window=window).mean()

def EMA(series, window):
    return series.ewm(span=window).mean()

def WMA(series, window):
    return series.rolling(window=window).apply(lambda x:(x*x/x.sum()).sum())

def DEMA(series, window):
    return 2*EMA(series, window)-EMA(EMA(series, window), window)

def DMA(series, N1, N2, M):
    """平行线差，是一种用来判断价格变化、买点卖点的指标体系。"""
    return  (series.rolling(N1).mean() - series.rolling(N2).mean()).rolling(M).mean()