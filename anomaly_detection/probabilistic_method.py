from show.plotly_show import *
from scipy.stats import skew
# 1、滑动窗口的3-sigma 【异常检测】
def sliding_3_sigma(win, t, a, ts):
    """
    滑动窗口的3-sigma
    :param win: windows，窗口大小
    :param t: 3-sigma ,t = 3
    :param a: 数值序列 pandas.core.series.Series
    :param ts: 时间序列 pandas.core.series.Series
    :return: x,y 异常点位置与数值
    """
    # 定义窗口大小和阈值
    window_size = win
    threshold = t

    # 检测异常值和突变
    anomalies = []
    for i in range(0,len(a) - window_size + 1):
        window = a[i:i+window_size]
        mean_w = np.mean(window)
        std_w = np.std(window)
        for j in range(window_size):
            if abs(window[j+i] - mean_w) > threshold * std_w:
                anomalies.append(i+j)
                # break

    # 显示异常点
    x = ts[anomalies]
    y = a[anomalies]

    # 可视化结果
    plotly_show(x,y,ts,a,"滑动窗口的3-sigma -- 异常检测")
    # matplotlib_show(x,y,ts,a,"滑动窗口的3-sigma -- 异常检测")

    return x,y

## 2、滑动窗口的箱线图 【异常检测】
#%%
def sliding_boxplot(win, t, a, ts):
    """
    滑动窗口的箱线图
    :param win: windows，窗口大小
    :param t: 阈值大小，阈值乘以IQR得到下边界(lower_bound)和上边界(upper_bound)
    :param a: 数值序列 pandas.core.series.Series
    :param ts: 时间序列 pandas.core.series.Series
    :return: x,y 异常点位置与数值
    """
    # 定义窗口大小和阈值
    window_size = win
    threshold = t

    # 检测异常值和突变
    anomalies = []
    for i in range(0, len(a) - window_size + 1):
        window = a[i:i + window_size]
        q1, q3 = np.percentile(window, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        for j in range(window_size):
            if window[j+i] < lower_bound or window[j+i] > upper_bound:
                anomalies.append(i + j)
                # break

    # 显示异常点
    x = ts[anomalies]
    y = a[anomalies]

    # 可视化结果
    plotly_show(x,y,ts,a,"滑动窗口的箱线图 -- 异常检测")  # matplotlib_show(x,y,ts,a,"滑动窗口的箱线图  -- 异常检测")

    return x,y

#%% md
## 3、滑动窗口 3_sigma 【数据修复】
#%%
def sliding_3_sigma_window_outlier_detection(win, t, a, ts, fill_threshold=True):
    """
    滑动窗口 3_sigma 数据修复
    :param win: windows，窗口大小
    :param t: 阈值大小，阈值乘以IQR得到下边界(lower_bound)和上边界(upper_bound)
    :param a: 数值序列 pandas.core.series.Series
    :param ts: 时间序列 pandas.core.series.Series
    :param fill_threshold:
    :return:
    """
    # 定义窗口大小和阈值
    window_size = win
    threshold = t
    # 检测异常值和突变
    anomalies = []
    for i in range(0, len(a) - window_size + 1):
        window = a[i:i+window_size]
        mean_w = np.mean(window)
        std_w = np.std(window)
        for j in range(window_size):
            if abs(window[j+i] - mean_w) > threshold * std_w:
                anomalies.append(i+j)
                # if fill_threshold: # 会改变原始数据
                #     a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w

    # # 可视化结果
    # fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    # # ax.scatter(ts[anomalies], a[anomalies], edgecolors='b', label='异常点')
    # ax.plot(ts, a, label='原始数据')
    #
    # # 填充阈值修复后的数据
    # if fill_threshold:
    #     fixed_a = a.copy()
    #     for i in range(len(a) - window_size + 1):
    #         window = a[i:i+window_size]
    #         mean_w = np.mean(window)
    #         std_w = np.std(window)
    #         for j in range(window_size):
    #             if abs(window[j+i] - mean_w) > threshold * std_w:
    #                 fixed_a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w
    #     ax.plot(ts, fixed_a, label='修复后数据')
    #
    # plt.title("滑动窗口 3_sigma+数据修复")
    # ax.set_xlabel('时间')
    # ax.set_ylabel('数值')
    # ax.legend()
    # fig.autofmt_xdate() # 自适应X坐标
    # plt.show()

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    # 创建画布
    fig = make_subplots(rows=1, cols=1)

    # 绘制原始数据
    fig.add_trace(go.Scatter(
        x=ts,
        y=a,
        name="原始数据"
    ))

    # 绘制修复后的数据
    if fill_threshold:
        fixed_a = a.copy()
        for i in range(len(a) - window_size + 1):
            window = a[i:i+window_size]
            mean_w = np.mean(window)
            std_w = np.std(window)
            for j in range(window_size):
                if abs(window[j+i] - mean_w) > threshold * std_w:
                    fixed_a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w
        fig.add_trace(go.Scatter(
            x=ts,
            y=fixed_a,
            name="修复后数据"
        ))

    # 设置图像标题和轴标签
    fig.update_layout(
        title="滑动窗口 3_sigma+数据修复",
        xaxis_title="时间",
        yaxis_title="数值",
        font=dict(family="Simhei"),
        width=1000,
        height=600,
        legend=dict(x=0.01, y=0.95),
        hovermode="x"
    )

    # 自适应X坐标
    fig.update_xaxes(rangemode="tozero", autorange=True)

    # 显示图像
    fig.show()

#%% md
## 4、滑动窗口 箱线图 【数据修复】
#%%
def sliding_boxplot_window_outlier_detection(win, t, a, ts, fill_threshold=True):
    """
    滑动窗口 箱线图 数据修复
    :param win: windows，窗口大小
    :param t: 阈值大小，阈值乘以IQR得到下边界(lower_bound)和上边界(upper_bound)
    :param a: 数值序列 pandas.core.series.Series
    :param ts: 时间序列 pandas.core.series.Series
    :param fill_threshold:
    :return:
    """
    # 定义窗口大小和阈值
    window_size = win
    threshold = t
    # 检测异常值和突变
    anomalies = []
    for i in range(0, len(a) - window_size + 1):
        window = a[i:i+window_size]
        q1, q3 = np.percentile(window, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        for j in range(window_size):
            if window[j+i] < lower_bound or window[j+i] > upper_bound:
                anomalies.append(i+j)
                # if fill_threshold:
                #     a[i+j] = lower_bound if window[j+i] < lower_bound else upper_bound

    # # 可视化结果
    # fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    # # ax.scatter(ts[anomalies], a[anomalies], edgecolors='b', label='异常点')
    # ax.plot(ts, a, label='原始数据')
    #
    # # 填充阈值修复后的数据
    # if fill_threshold:
    #     fixed_a = a.copy()
    #     for i in range(len(a) - window_size + 1):
    #         window = a[i:i+window_size]
    #         q1, q3 = np.percentile(window, [25, 75])
    #         iqr = q3 - q1
    #         lower_bound = q1 - threshold * iqr
    #         upper_bound = q3 + threshold * iqr
    #         for j in range(window_size):
    #             if window[j+i] < lower_bound or window[j+i] > upper_bound:
    #                 fixed_a[i+j] = lower_bound if window[j+i] < lower_bound else upper_bound
    #     ax.plot(ts, fixed_a, label='修复后数据')
    #
    # plt.title("滑动窗口 箱线图+数据修复")
    # ax.set_xlabel('时间')
    # ax.set_ylabel('数值')
    # ax.legend()
    # fig.autofmt_xdate() # 自适应X坐标
    # plt.show()
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    # 绘制原始数据
    trace1 = go.Scatter(x=ts, y=a, name='原始数据')

    # 填充阈值修复后的数据
    if fill_threshold:
        fixed_a = a.copy()
        for i in range(len(a) - window_size + 1):
            window = a[i:i+window_size]
            q1, q3 = np.percentile(window, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            for j in range(window_size):
                if window[j+i] < lower_bound or window[j+i] > upper_bound:
                    fixed_a[i+j] = lower_bound if window[j+i] < lower_bound else upper_bound
        trace2 = go.Scatter(x=ts, y=fixed_a, name='修复后数据')
    else:
        trace2 = None

    # 创建子图
    fig = make_subplots(rows=1, cols=1)

    # 添加数据到子图
    fig.add_trace(trace1)
    if trace2:
        fig.add_trace(trace2)

    # 更新子图布局
    fig.update_layout(
        title="滑动窗口 箱线图+数据修复",
        xaxis_title="时间",
        yaxis_title="数值",
        font=dict(family="Simhei"),
        width=1000,
        height=600
    )

    # 自适应X坐标
    fig.update_xaxes(automargin=True)

    # 显示图像
    fig.show()

#%% md
## 5、滑动窗口 偏度+3_sigma 【数据修复】
#%%
def sliding_window_skew_3_sigma_outlier_detection(win, t, a, ts, fill_threshold=True):
    """
    滑动窗口 偏度+3_sigma 数据修复
    :param win: windows，窗口大小
    :param t: 阈值大小，阈值乘以IQR得到下边界(lower_bound)和上边界(upper_bound)
    :param a: 数值序列 pandas.core.series.Series
    :param ts: 时间序列 pandas.core.series.Series
    :param fill_threshold:
    :return:
    """
    # 定义窗口大小和阈值
    window_size = win
    threshold = t

    # 检测异常值和突变
    anomalies = []
    for i in range(0, len(a) - window_size + 1):
        window = a[i:i+window_size]
        mean_w = np.mean(window)
        std_w = np.std(window)
        skewness_w = skew(window)  # 计算窗口偏度
        lower_bound = mean_w - threshold * std_w  # 计算下边界
        upper_bound = mean_w + threshold * std_w  # 计算上边界

        for j in range(window_size):
            if (window[j+i] < lower_bound or window[j+i] > upper_bound) or abs(skewness_w) > 1.8:  # 如果超出边界或偏度绝对值大于1
                anomalies.append(i+j)
                # if fill_threshold:
                #     a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w

    # # 可视化结果
    # fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    # # ax.scatter(ts[anomalies], a[anomalies], edgecolors='b', label='异常点')
    # ax.plot(ts, a, label='原始数据')
    #
    # # 填充阈值修复后的数据
    # if fill_threshold:
    #     fixed_a = a.copy()
    #     for i in range(len(a) - window_size + 1):
    #         window = a[i:i+window_size]
    #         mean_w = np.mean(window)
    #         std_w = np.std(window)
    #         skewness_w = skew(window)  # 计算窗口偏度
    #         for j in range(window_size):
    #             if abs(window[j+i] - mean_w) > threshold * std_w  or abs(skewness_w) > 1.8:
    #                 fixed_a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w
    #     ax.plot(ts, fixed_a, label='修复后数据')
    #
    # plt.title("滑动窗口 偏度+3_sigma 数据修复")
    # ax.set_xlabel('时间')
    # ax.set_ylabel('数值')
    # ax.legend()
    # fig.autofmt_xdate() # 自适应X坐标
    # plt.show()
    import plotly.graph_objects as go

    # 填充阈值修复后的数据
    if fill_threshold:
        fixed_a = a.copy()
        for i in range(len(a) - window_size + 1):
            window = a[i:i+window_size]
            mean_w = np.mean(window)
            std_w = np.std(window)
            skewness_w = skew(window)  # 计算窗口偏度
            for j in range(window_size):
                if abs(window[j+i] - mean_w) > threshold * std_w or abs(skewness_w) > 1.8:
                    fixed_a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w

    # 绘制原始数据和修复后的数据
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=a, mode='lines', name='原始数据'))
    if fill_threshold:
        fig.add_trace(go.Scatter(x=ts, y=fixed_a, mode='lines', name='修复后数据'))

    # 设置图形属性
    fig.update_layout(title="滑动窗口 偏度+3_sigma 数据修复",
                      xaxis_title='时间', yaxis_title='数值',
                      legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02),
                      width=1000, height=600, template='simple_white')

    # 自适应X坐标
    fig.update_xaxes(rangeslider=dict(visible=True),
                     rangeselector=dict(buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])))
    fig.show()


#%% md
## 6、滑动窗口 偏度+箱线图【数据修复】
#%%
def sliding_window_skew_boxplot_outlier_detection(win, t, a, ts, fill_threshold=True):
    """
    滑动窗口 偏度+箱线图 数据修复
    :param win: windows，窗口大小
    :param t: 阈值大小，阈值乘以IQR得到下边界(lower_bound)和上边界(upper_bound)
    :param a: 数值序列 pandas.core.series.Series
    :param ts: 时间序列 pandas.core.series.Series
    :param fill_threshold: 是否对异常值进行修复
    :return:
    """
    window_size = win
    threshold = t

    # 检测异常值和突变
    anomalies = []
    for i in range(0, len(a) - window_size + 1):
        window = a[i:i + window_size]
        mean_w = np.mean(window)
        std_w = np.std(window)
        skewness = ((window - mean_w) ** 3).sum() / (len(window) * std_w ** 3)
        Q1 = np.percentile(window, 25)
        Q3 = np.percentile(window, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        for j in range(window_size):
            if window[j+i] < lower_bound or window[j+i] > upper_bound or abs(skewness) > 2:
                anomalies.append(i+j)
                # if fill_threshold:
                #     a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w

    # # 可视化结果
    # fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    # # ax.scatter(ts[anomalies], a[anomalies], edgecolors='b', label='异常点')
    # ax.plot(ts, a, label='原始数据')
    #
    # # 填充阈值修复后的数据
    # if fill_threshold:
    #     fixed_a = a.copy()
    #     for i in range(len(a) - window_size + 1):
    #         window = a[i:i+window_size]
    #         mean_w = np.mean(window)
    #         std_w = np.std(window)
    #         skewness = ((window - mean_w) ** 3).sum() / (len(window) * std_w ** 3)
    #         Q1 = np.percentile(window, 25)
    #         Q3 = np.percentile(window, 75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - threshold * IQR
    #         upper_bound = Q3 + threshold * IQR
    #         for j in range(window_size):
    #             if window[j+i] < lower_bound or window[j+i] > upper_bound or abs(skewness) > 2:
    #                 fixed_a[i+j] = lower_bound if window[j+i] < lower_bound else upper_bound
    #                 # fixed_a[i+j] = mean_w + threshold * std_w if window[j+i] > mean_w else mean_w - threshold * std_w
    #     ax.plot(ts, fixed_a, label='修复后数据')
    #
    # plt.title("滑动窗口 偏度+箱线图 数据修复")
    # ax.set_xlabel('时间')
    # ax.set_ylabel('数值')
    # ax.legend()
    # fig.autofmt_xdate() # 自适应X坐标
    # plt.show()
    import plotly.graph_objects as go

    # 填充阈值修复后的数据
    if fill_threshold:
        fixed_a = a.copy()
        for i in range(len(a) - window_size + 1):
            window = a[i:i+window_size]
            mean_w = np.mean(window)
            std_w = np.std(window)
            skewness = ((window - mean_w) ** 3).sum() / (len(window) * std_w ** 3)
            Q1 = np.percentile(window, 25)
            Q3 = np.percentile(window, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for j in range(window_size):
                if window[j+i] < lower_bound or window[j+i] > upper_bound or abs(skewness) > 2:
                    fixed_a[i+j] = lower_bound if window[j+i] < lower_bound else upper_bound

    # 绘制原始数据和修复后的数据
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=a, mode='lines', name='原始数据'))
    if fill_threshold:
        fig.add_trace(go.Scatter(x=ts, y=fixed_a, mode='lines', name='修复后数据'))

    # 设置图形属性
    fig.update_layout(title="滑动窗口 偏度+箱线图 数据修复",
                      xaxis_title='时间', yaxis_title='数值',
                      legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02),
                      width=1000, height=600, template='simple_white')

    # 自适应X坐标
    fig.update_xaxes(rangeslider=dict(visible=True),
                     rangeselector=dict(buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])))
    fig.show()

## 高维滑动窗口的3-sigma 【异常检测】
# def sliding_3_sigma_high(win, t, val, ts, name):
#     """
#     滑动窗口的3-sigma
#     :param win: windows，窗口大小
#     :param t: 3-sigma ,t = 3
#     :param val: 数值序列数组 pandas.core.series.Series 'acceleration_peak_x' ,'acceleration_rms_x','speed_peak_x','speed_rms_x'
#     :param ts: 时间序列 pandas.core.series.Series
#     :return: x,y 异常点位置与数值
#     """
#     # 定义窗口大小和阈值
#     window_size = win
#     threshold = t
#
#     plt.rcParams['font.sans-serif'] = 'Simhei' # 显示中文
#     plt.figure(figsize=(16,10),dpi=150)
#
#     # 异常相关度记录，大于2就整体异常
#     # 空间复杂度换时间复杂度
#     Relevance = np.zeros(len(val[0]))
#
#     MAX_Y = 0
#     # 检测异常值和突变
#     for a in val:
#         print(a.name)
#         anomaly = []
#         for i in range(0,len(a) - window_size + 1):
#             window = a[i:i+window_size]
#             mean_w = np.mean(window)
#             std_w = np.std(window)
#             for j in range(window_size):
#                 if abs(window[j+i] - mean_w) > threshold * std_w:
#                     anomaly.append(i+j)
#                     # break
#
#         # 记录相关度
#         Relevance[anomaly] = Relevance[anomaly] + 1
#
#         # 显示异常点
#         x = ts[anomaly]
#         y = a[anomaly]
#
#         # 获得y轴最大值 MAX_Y
#         MAX_Y = max(y.max(),MAX_Y)
#         print(MAX_Y)
#
#         # 可视化结果
#         plt.plot(ts,a,label = a.name )
#         plt.scatter(x, y , c ='r')
#
#
#     # 统计相关度大于R的点
#     R_anomaly = []
#     R = 2
#     for i in range(0,len(Relevance)):
#         if Relevance[i] > R:
#             R_anomaly.append(i)
#
#     # 显示异常点
#     X = ts[R_anomaly]
#     # R_anomaly2 =[ R_anomaly[i] + 1 for i in range(0,len(R_anomaly))]
#     # X1 = ts[R_anomaly2]
#
#     # 可视化结果
#     print("相关性>",R,"的异常点：\n",X)
#     plt.fill_between ( x = X, y1=0,y2=MAX_Y+1,facecolor='green', alpha=0.3 ,label = "相关度较大的异常点")
#
#     plt.gcf().autofmt_xdate() # 自适应X坐标
#     plt.title("滑动窗口的3-sigma -- 高维异常检测 -- "+name)
#     plt.xlabel('时间')
#     plt.ylabel('数值')
#     plt.legend()  # 显示图例
#     plt.show()
#%%
import plotly.io as pio
import pandas as pd
import numpy as np

## 高维滑动窗口的3-sigma 【异常检测】
def sliding_3_sigma_high(win, t, val, ts, name):
    """
    滑动窗口的3-sigma
    :param win: windows，窗口大小
    :param t: 3-sigma ,t = 3
    :param val: 数值序列数组 pandas.core.series.Series 'acceleration_peak_x' ,'acceleration_rms_x','speed_peak_x','speed_rms_x'
    :param ts: 时间序列 pandas.core.series.Series
    :return: None
    """
    # 定义窗口大小和阈值
    window_size = win
    threshold = t

    fig = go.Figure()

    # 异常相关度记录，大于2就整体异常
    # 空间复杂度换时间复杂度
    Relevance = np.zeros(len(val[0]))

    MAX_Y = 0
    # 检测异常值和突变
    for a in val:
        print(a.name)
        anomaly = []
        for i in range(0,len(a) - window_size + 1):
            window = a[i:i+window_size]
            mean_w = np.mean(window)
            std_w = np.std(window)
            for j in range(window_size):
                if abs(window[j+i] - mean_w) > threshold * std_w:
                    anomaly.append(i+j)
                    # break

        # 记录相关度
        Relevance[anomaly] = Relevance[anomaly] + 1

        # 显示异常点
        x = ts[anomaly]
        y = a[anomaly]

        # 获得y轴最大值 MAX_Y
        MAX_Y = max(y.max(),MAX_Y)
        print(MAX_Y)

        # 可视化结果
        fig.add_trace(go.Scatter(x=ts, y=a, mode='lines', name=a.name))
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='异常点'))

    # 统计相关度大于R的点
    R_anomaly = []
    R = 3
    for i in range(0,len(Relevance)):
        if Relevance[i] > R:
            R_anomaly.append(i)

    # 显示异常点
    X = ts[R_anomaly]

    # 可视化结果
    print("相关性>",R,"的异常点：\n",X)
    # fig.add_trace(go.Scatter(x=X, y=np.zeros(len(X)), mode='markers', fill='tonexty', fillcolor='green', line=dict(width=1), name="相关度较大的异常点"))
    if X.shape[0]>0 :
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=X.iloc[0],
            y0=0,
            x1=X.iloc[-1],
            y1=1,
            fillcolor='green',
            opacity=0.3,
            layer="below",
        )
    fig.update_layout(title=f"滑动窗口的3-sigma -- 高维异常检测 -- {name}", xaxis_title="时间", yaxis_title="数值",width=1000, height=600 , legend_title="数据序列")
    fig.update_xaxes(rangeslider_visible=True)
    pio.show(fig)

def sliding_3_sigma_return(win, t, a, ts):
    """
    滑动窗口的3-sigma
    :param win: windows，窗口大小
    :param t: 3-sigma ,t = 3
    :param a: 数值序列 pandas.core.series.Series
    :param ts: 时间序列 pandas.core.series.Series
    :return: lower, upper 阈值
    """
    # 定义窗口大小和阈值
    window_size = win
    threshold = t

    # 检测异常值和突变
    anomalies = []
    for i in range(0,len(a) - window_size + 1):
        window = a[i:i+window_size]
        mean_w = np.mean(window)
        std_w = np.std(window)
        for j in range(window_size):
            if abs(window[j+i] - mean_w) > threshold * std_w:
                anomalies.append(i+j)
                # break

    # 显示异常点
    x = ts[anomalies]
    y = a[anomalies]

    # 可视化结果
    plotly_show(x,y,ts,a,"滑动窗口的3-sigma -- 异常检测")
    # matplotlib_show(x,y,ts,a,"滑动窗口的3-sigma -- 异常检测")

    mean = np.mean(a)
    std = np.std(a)
    lower = mean - t * std
    upper = mean + t * std
    return lower, upper