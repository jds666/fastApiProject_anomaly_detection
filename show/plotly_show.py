import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# 绘制散点图和折线图
def plotly_show(x, y, ts, a, tl):
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(
        go.Scatter(x=x, y=y, mode='markers', marker=dict(color='blue'), name='异常点'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=ts, y=a, mode='lines', name='数值折线'),
        secondary_y=False,
    )

    # 设置图形布局和样式
    fig.update_layout(
        title=tl,
        xaxis_title="时间",
        yaxis_title="数值",
        font=dict(family='Simhei'),
        width=1000,
        height=600,
    )

    # 显示图形
    fig.show()

