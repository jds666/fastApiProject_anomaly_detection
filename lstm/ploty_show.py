import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt


def plot_show_plotly(original_data,anomalies,datetime,train_size,id = " "):

    # 绘制每个特征的曲线
    fig = go.Figure()
    for i in range(original_data.shape[1]):
        trace = go.Scatter(x=datetime,#list(range(len(original_data))),
                           y=original_data[:, i],
                           mode='lines',
                           name=f'Feature {i+1}')
        fig.add_trace(trace)

    # 绘制异常点散点图
    anomalies_idx = train_size + anomalies[0]

    #单维有异常点，多维是异常区域
    if original_data.shape[1] == 1:
        trace2 = go.Scatter(x=np.array(datetime)[anomalies_idx], # anomalies_idx,
                            y=np.squeeze(original_data[anomalies, 0]),
                            mode='markers',
                            marker=dict(color='red', size=6),
                            name='Anomalies')
        fig.add_trace(trace2)


    # 绘制异常点阴影区域
    x_data = []

    for i in range(len(anomalies_idx)):
        x_data.append(anomalies_idx[i])

        # Check if the next point is continuous
        if i < len(anomalies_idx) - 1 and anomalies_idx[i + 1] - anomalies_idx[i] > 1:
            # If the next point is not continuous, draw a rectangle
            fig.add_shape(type='rect',
                          x0=datetime[x_data[0]],#x_data[0] - 0.5,
                          y0=np.min(original_data[:,:]),
                          x1=datetime[x_data[-1]],#x_data[-1] + 0.5,
                          y1=np.max(original_data[:,:]),
                          fillcolor='rgba(255, 0, 0, 0.3)',
                          line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                          opacity=0.5)
            x_data = []

    # 最后一个区间上色
    if x_data:
        fig.add_shape(type='rect',
                      x0=datetime[x_data[0]],#x_data[0] - 0.5,
                      y0=np.min(original_data[:, :]),
                      x1=datetime[x_data[-1]],#x_data[-1] + 0.5,
                      y1=np.max(original_data[:, :]),
                      fillcolor='rgba(255, 0, 0, 0.3)',
                      line=dict(color='rgba(255, 0, 0, 0.3)', width=0.2),
                      opacity=0.5)



    # 绘制图表
    fig.update_layout(title='Original Data and Anomalies      '+id,
                      xaxis_title='Time',
                      yaxis_title='Value',
                      showlegend=True)

    fig.show()
    return fig

def plot_show_plotly_repair(original_data,repair_data,datetime,id = " "):
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
                      showlegend=True)

    fig.show()
    return fig


def plot_show_matplot(original_data,anomalies_data,anomalies,train_size):
    # 绘制原始数据和异常点
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data')
    plt.plot(train_size + anomalies[0], anomalies_data, 'ro', label='Anomalies')
    plt.legend()
    plt.show()
    return plt