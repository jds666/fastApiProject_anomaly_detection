import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from lof.plotly_show import *
from lstm.utils import *

config = get_config_from_json(json_file)
def JsonAnomalyDetectionTemperatureLof(Data):

    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})
    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")
    # 定义特征列
    features = ['values']
    # 进行异常检测
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred = clf.fit_predict(data[features])
    data['outlier'] = y_pred

    data['outlier'] = data['outlier'].replace(1, 0).replace(-1, 1)

    # 绘制折线图，其中红色点表示异常值
    if config["is_plot_result"]:
        plot_show_plotly_lof(data,Data.id)
    return data['outlier'].tolist()

def JsonAnomalyDetectionVibrationLof(Data):
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)
    # data timestamps 转换为日期格式
    data.index = pd.to_datetime(data.index.values, unit="ms")
    # 定义特征列
    features = Data.valueNameList
    # 进行异常检测
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred = clf.fit_predict(data[features])
    # print(y_pred)
    # # 输出异常分数
    # scores = -clf.negative_outlier_factor_+1
    # print(scores)
    data['outlier'] = y_pred

    data['outlier'] = data['outlier'].replace(1, 0).replace(-1, 1)

    # 绘制折线图，其中红色区域表示异常值
    if config["is_plot_result"]:
        plot_show_plotly_vibration_lof(data,Data.id)
    return data['outlier'].to_list()

def JsonRepairTemperatureLof(Data):

    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})

    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")
    data_time = [str(ts) for ts in data.index.tolist()]
    # 定义特征列
    features = ['values']
    # 进行异常检测
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred = clf.fit_predict(data[features])

    data['outlier'] = y_pred

    data['outlier'] = data['outlier'].replace(1, 0).replace(-1, 1)

    # 使用线性插值修复异常值
    repaired_values = data['values'].copy()
    repaired_values[data['outlier'] == 1] = np.nan
    repaired_values = repaired_values.interpolate(method='linear')
    repaired_values = repaired_values.fillna(method='ffill').fillna(method='bfill')
    repairData = repaired_values.tolist()

    # 绘图，包括原始数据与修复数据
    if config["is_plot_result"]:
        plot_show_plotly_repair_lof(data, repaired_values, Data.id )

    return data['outlier'].to_list(), repairData
