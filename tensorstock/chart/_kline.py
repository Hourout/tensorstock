from pyecharts import options as opts
from pyecharts.charts import Kline

__all__ = ['KLine']

class KLine():
    def __init__(self):
        pass
    
    def render_notebook(self):
        return self.c.render_notebook()
    
    def render_html(self, path="kline.html"):
        return self.c.render(path)
        
    def kline(self, x_data, y_data, name="kline", title="Kline"):
        self.c = (
            Kline()
            .add_xaxis(x_data)
            .add_yaxis(name, y_data, 
#                        markline_opts=opts.MarkLineOpts(
#             data=[opts.MarkLineItem(type_="max", value_dim="open")]),
                      )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True, 
                                         splitarea_opts=opts.SplitAreaOpts(
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                datazoom_opts=[opts.DataZoomOpts()],
                title_opts=opts.TitleOpts(title=title),
            )
        )
        return self