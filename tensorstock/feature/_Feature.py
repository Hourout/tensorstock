from tensorstock.feature._Feature_config import *

__all__ = ['Feature']

class Feature():
    def __init__(self, df):
        self.feature = df
        self._feature(self.feature)
        self.columns_translation = columns_translation
        self.columns_explain = columns_explain
        
    def _feature(self, df):
        df = df.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'highest', '最低':'lowest', 
         '成交量':'volume', '成交额':'turnover', '振幅':'amplitude', '涨跌幅':'change_rate', 
         '涨跌额':'change_amount', '换手率':'turnover_rate'})
        df['solid_rate'] = (df.close-df.open)/(df.highest-df.lowest)
        df['shadow_down_rate'] = (df[['open', 'close']].min(axis=1)-df.lowest)/(df.highest-df.lowest)
        df['shadow_up_rate'] = (df.highest-df[['open', 'close']].max(axis=1))/(df.highest-df.lowest)
