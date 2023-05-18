import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

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


# 单维修复
def plot_show_single_repair(original_data, repair_data, datetime, id=" "):
    '''
    Plot the data 单维修复.
    :param original_data: pandas
    :param repair_data: pandas
    :param datetime: pandas
    :param id: int
    :return: fig
    '''
    # 绘制每个特征的曲线
    fig = go.Figure()
    # 原始
    trace = go.Scatter(x=datetime,
                       y=original_data,
                       mode='lines',
                       name=f'original_data ')
    fig.add_trace(trace)
    # 修复
    trace2 = go.Scatter(x=datetime,
                        y=repair_data,
                        mode='lines',
                        name=f'repair_data ')
    fig.add_trace(trace2)

    # 绘制图表
    fig.update_layout(title='Original Data and Repaired Data      ' + id,
                      xaxis_title='Time',
                      yaxis_title='Value',
                      font=dict(size=18),
                      showlegend=True)

    fig.show()
    return fig

#高维修复

def plot_show_Mult_repair(original_data,repair_data,datetime,id = " "):
    '''
    Plot the data 高维修复.
    :param original_data:
    :param repair_data:
    :param datetime:
    :param id:
    :return:
    '''
    # 绘制每个特征的曲线
    fig = go.Figure()
    for i in range(original_data.shape[1]):
        #原始
        trace = go.Scatter(x=datetime,
                           y=original_data[:, i],
                           mode='lines',
                           name=f'original_data {i + 1}')
        fig.add_trace(trace)
        #修复
        trace2 = go.Scatter(x=datetime,
                           y=repair_data[:, i],
                           mode='lines',
                           name=f'repair_data {i + 1}')
        fig.add_trace(trace2)

    # 绘制图表
    fig.update_layout(title='Original Data and Repaired Data      ' + id,
                      xaxis_title='Time',
                      yaxis_title='Value',
                      font=dict(size=18),
                      showlegend=True)

    fig.show()
    return fig

#异常检测绘图
def plot_show_plotly(original_data,anomalies,datetime,id = " "):
    '''
    Plot the data using plotly.
    :param original_data: List
    :param anomalies: List
    :param datetime:pandas
    :param id:int
    :return:
    '''
    print(datetime)
    print(original_data)
    # 绘制每个特征的曲线
    fig = go.Figure()
    if  original_data.shape[1] == 1:
        trace = go.Scatter(x=datetime,
                           y=original_data.iloc[:, 0],
                           mode='lines',
                           name=original_data.columns.tolist()[0])
        fig.add_trace(trace)
        trace2 = go.Scatter(x=[datetime[idx] for idx, val in enumerate(anomalies) if val == 1],
                            y=[original_data.iloc[idx, 0] for idx, val in enumerate(anomalies) if val == 1],
                            mode='markers',
                            marker=dict(color='red', size=6),
                            name='Anomalies')
        fig.add_trace(trace2)
    else:
        for col_label, col_data in original_data.iteritems():
            trace = go.Scatter(x=datetime,
                               y=col_data,
                               mode='lines',
                               name=col_label)
            fig.add_trace(trace)

        print("anomalies:",anomalies)
        anomalies_idx = [idx for idx, val in enumerate(anomalies) if val == 1]
        print("anomalies_idx:",anomalies_idx)
        # 绘制异常点阴影区域
        x_data = []

        for i in range(len(anomalies_idx)):
            x_data.append(anomalies_idx[i])

            # Check if the next point is continuous
            if i < len(anomalies_idx) - 1 and anomalies_idx[i + 1] - anomalies_idx[i] > 1:
                # If the next point is not continuous, draw a rectangle
                fig.add_shape(type='rect',
                              x0=datetime[int(x_data[0] - 0.5)],#x_data[0] - 0.5,
                              y0=original_data.iloc[:,1:].min().min(),
                              x1=datetime[int(x_data[-1] + 0.5)],#x_data[-1] + 0.5,
                              y1=original_data.iloc[:,1:].max().max(),
                              fillcolor='rgba(255, 0, 0, 0.3)',
                              line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                              opacity=0.5)
                x_data = []

        # 最后一个区间上色
        if x_data:
            fig.add_shape(type='rect',
                          x0=datetime[int(x_data[0] - 0.5)],#x_data[0] - 0.5,
                          y0=original_data.iloc[:,1:].min().min(),
                          x1=datetime[int(x_data[-1] + 0.5)],#x_data[-1] + 0.5,
                          y1=original_data.iloc[:,1:].max().max(),
                          fillcolor='rgba(255, 0, 0, 0.3)',
                          line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                          opacity=0.5,
                          name='Contiguous Anomalies')

    # 绘制图表
    fig.update_layout(title='Original Data and Anomalies      '+id,
                      xaxis_title='Time',
                      yaxis_title='Value',
                      font=dict(size=18),
                      showlegend=True)

    fig.show()
    return fig
