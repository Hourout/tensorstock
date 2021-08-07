__all__ = ['MACD', 'KDJ', 'BIAS', 'BBANDS', 'RSI', 'WR']

def MACD(series, short=12, long=26, mid=9):
    ema_short = series.ewm(adjust=False, span=short, ignore_na=True).mean()
    ema_long = series.ewm(adjust=False, span=long, ignore_na=True).mean()
    ema_diff = (ema_short-ema_long)
    ema_dea = ema_diff.ewm(adjust=False, span=mid, ignore_na=True).mean()
    macd = 2*(ema_diff-ema_dea)
    return macd

def KDJ(data, N=9, M=2):
    lowList = data['low'].rolling(N).min().fillna(value=data['low'].expanding().min())
    highList = data['high'].rolling(N).max().fillna(value=data['high'].expanding().max())
    rsv = (data['close'] - lowList) / (highList - lowList) * 100
    kdj_k = rsv.ewm(alpha=1/M, adjust=False).mean()
    kdj_d = kdj_k.ewm(alpha=1/M, adjust=False).mean()
    kdj_j = 3.0 * kdj_k - 2.0 * kdj_d
    return {'kdj_k':kdj_k, 'kdj_d':kdj_d, 'kdj_j':kdj_j}

def BIAS(series, N1, N2, N3):
    """乖离率，描述收盘价距离均线的百分比，常用来衡量收盘价偏离程度。"""
    bias1 = (series - series.rolling(N1).mean())/series.rolling(N1).mean() * 100
    bias2 = (series - series.rolling(N2).mean())/series.rolling(N2).mean() * 100
    bias3 = (series - series.rolling(N3).mean())/series.rolling(N3).mean() * 100
    return {'bias1':bias1, 'bias2':bias2, 'bias3':bias3}

def BBANDS(series, window):
    middleband = series.rolling(window).mean()
    upperband = middleband + 2 * series.rolling(window).std()
    lowerband = middleband - 2 * series.rolling(window).std()
    return {'middleband':middleband, 'upperband':upperband, 'lowerband':lowerband}

def RSI(series, N1, N2, N3):
    ''' 计算RSI相对强弱指数'''
    diff = series.diff().fillna(0)
    x = diff.clip(lower=0)
    rsi1 = x.ewm(alpha=1 / N1, adjust=False).mean() / (diff.abs().ewm(alpha=1/N1, adjust=False).mean()) * 100
    rsi2 = x.ewm(alpha=1 / N2, adjust=False).mean() / (diff.abs().ewm(alpha=1 / N2, adjust=False).mean()) * 100
    rsi3 = x.ewm(alpha=1 / N3, adjust=False).mean() / (diff.abs().ewm(alpha=1 / N3, adjust=False).mean()) * 100
    return {'rsi1':rsi1, 'rsi2':rsi2, 'rsi3':rsi3}

def WR(data, window):
    '''计算威廉指数'''
    a = data['high'].rolling(window).max() - data['close']
    b = data['high'].rolling(window).max() - data['low'].rolling(window).min()
    c = data['high'].expanding().max() - data['close']
    d = data['high'].expanding().max() - data['low'].expanding().min()
    wr = (a/b).fillna(c/d)*100
    return wr