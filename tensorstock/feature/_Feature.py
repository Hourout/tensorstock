from tensorstock.feature._Feature_config import *

__all__ = ['Feature']

class Feature():
    def __init__(self, df):
        self.feature = self._feature(df)
        self.columns_translation = columns_translation
        self.columns_explain = columns_explain
        
    def _feature(self, df):
        df = df.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'highest', '最低':'lowest', 
         '成交量':'volume', '成交额':'turnover', '振幅':'amplitude', '收盘涨跌幅':'change_close_rate', 
         '涨跌额':'change_amount', '换手率':'turnover_rate'})
        df['solid_rate'] = (df.close-df.open)/(df.highest-df.lowest)
        df['shadow_down_rate'] = (df[['open', 'close']].min(axis=1)-df.lowest)/(df.highest-df.lowest)
        df['shadow_up_rate'] = (df.highest-df[['open', 'close']].max(axis=1))/(df.highest-df.lowest)
        df['momentum_up_rate'] = ((df.highest-(df.close-df.change_amount)).clip(lower=0)/(df.highest-df.lowest)).clip(upper=1).round(3)
        df['centroid_true_price'] = (df.close+df.open)/2
        df['centroid_false_price'] = (df.highest+df.lowest)/2
        df['change_open_rate'] = (df.open-(df.close-df.change_amount))/(df.close-df.change_amount)
        df['change_highest_rate'] = (df.highest-(df.close-df.change_amount))/(df.close-df.change_amount)
        df['change_lowest_rate'] = (df.lowest-(df.close-df.change_amount))/(df.close-df.change_amount)
        return df
