import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
# 绘制折线图，其中红色点表示异常值
def plot_show_plotly_lof(data,id):
    fig = go.Figure()
    trace1 = go.Scatter(x=data["timestamps"], y=data["values"], mode='lines', name='actual')
    trace2 = go.Scatter(x=data.loc[data['outlier'] == 1, ["timestamps"]].timestamps.tolist(),
                        y=np.squeeze(data.loc[data['outlier'] == 1, ["values"]].values).tolist(),
                        mode='markers',
                        marker=dict(color='red', size=6), name='outlier')
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    # 绘制图表
    fig.update_layout(title='Original Data and Anomalies      ' + id,
                      xaxis_title='Time',
                      yaxis_title='Value',
                      font=dict(size=18),
                      showlegend=True)
    fig.show()
    return fig

def plot_show_plotly_vibration_lof(data, id):

    # 绘制每个特征的曲线
    fig = go.Figure()
    #print(data.shape[1]-2)
    for i in range(data.shape[1]-1):
        #print(data.columns[i])
        trace = go.Scatter(x=data.index,#list(range(len(original_data))),
                           y=data.iloc[:, i],
                           mode='lines',
                           name=f'Feature {i+1}')
        fig.add_trace(trace)

    # 绘制异常点散点图
    anomalies_idx = np.where(data["outlier"] == 1)[0].tolist()#data[data["outlier"] == 1].index.tolist()
    #print(anomalies_idx)

    # 绘制异常点阴影区域
    x_data = []

    for i in range(len(anomalies_idx)):
        x_data.append(anomalies_idx[i])
        #print(x_data)

        # Check if the next point is continuous
        if i < len(anomalies_idx) - 1 and anomalies_idx[i + 1] - anomalies_idx[i] > 1:
            # If the next point is not continuous, draw a rectangle
            fig.add_shape(type='rect',
                          x0=data.index[x_data[0]],#x_data[0] - 0.5,
                          y0=np.min(data.values),
                          x1=data.index[x_data[-1]],#x_data[-1] + 0.5,
                          y1=np.max(data.values),
                          fillcolor='rgba(255, 0, 0, 0.3)',
                          line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                          opacity=0.5)
            x_data = []

    # 最后一个区间上色
    if x_data:
        fig.add_shape(type='rect',
                      x0=data.index[x_data[0]],  # x_data[0] - 0.5,
                      y0=np.min(data.values),
                      x1=data.index[x_data[-1]],  # x_data[-1] + 0.5,
                      y1=np.max(data.values),
                      fillcolor='rgba(255, 0, 0, 0.3)',
                      line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                      opacity=0.5)



    # 绘制图表
    fig.update_layout(title='Original Data and Anomalies      '+id,
                      xaxis_title='Time',
                      yaxis_title='Value',
                      font=dict(size=18),
                      showlegend=True)

    fig.show()
    return fig


def plot_show_plotly_repair_lof(data, repaired_values, id ):
    # 绘制每个特征的曲线
    fig = go.Figure()

    # 原始
    trace = go.Scatter(x=data["timestamps"],
                       y=data['values'],
                       mode='lines',
                       name=f'original_data ')
    fig.add_trace(trace)
    # 修复
    trace2 = go.Scatter(x=data["timestamps"],
                        y=repaired_values.values,
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