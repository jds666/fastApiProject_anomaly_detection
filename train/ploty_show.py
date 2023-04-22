import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt


def plot_show_plotly(original_data,anomalies_data,anomalies,train_size):

    # 绘制每个特征的曲线
    fig = go.Figure()
    for i in range(original_data.shape[1]):
        trace = go.Scatter(x=list(range(len(original_data))),
                           y=original_data[:, i],
                           mode='lines',
                           name=f'Feature {i+1}')
        fig.add_trace(trace)

    # 绘制异常点曲线
    anomalies_idx = train_size + anomalies[0]

    trace2 = go.Scatter(x=anomalies_idx,
                        y=anomalies_data[:, 0],
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name='Anomalies')
    fig.add_trace(trace2)

    # 绘制图表
    fig.update_layout(title='Original Data and Anomalies',
                      xaxis_title='Index',
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