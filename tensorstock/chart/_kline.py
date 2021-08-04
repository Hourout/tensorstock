from pyecharts import options as opts
from pyecharts.charts import Kline

__all__ = ['KLine']

class KLine():
    def __init__(self, title="Kline"):
        self._c = (
            Kline()
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True, 
                                         splitarea_opts=opts.SplitAreaOpts(
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                datazoom_opts=[opts.DataZoomOpts(range_start=40, range_end=100)],
                title_opts=opts.TitleOpts(title=title),
            )
        )
        
    def render_notebook(self):
        return self._c.render_notebook()
    
    def render_html(self, path="kline.html"):
        return self._c.render(path)
        
    def add_xaxis(self, x_data):
        self._c.add_xaxis(x_data)
        return self
    
    def add_yaxis(self, y_data, name="kline"):
        self._c.add_yaxis(name, y_data)
        return self