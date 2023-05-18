import pandas as pd
import numpy as np
from lstm.utils import *
from show.plotly_show import *

config = get_config_from_json(json_file)

def JsonAnomalyDetectionTemperatureBoxplot(Data, k):
    # json to pandas
    #Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列
    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})
    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")

    # 计算上下四分位数
    q1, q3 = np.percentile(data['values'], [25, 75])
    iqr = q3 - q1

    # 计算离群值的范围
    upper_bound = q3 + k * iqr
    lower_bound = q1 - k * iqr

    # 标记异常值,如果数据小于下限或大于上限则为1，反之为0
    anomaly_label = []
    for value in data["values"]:
        if (value < lower_bound).any() or (value > upper_bound).any():
            # print(value)
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)


    # 绘制折线图，其中红色点表示异常值
    if config["is_plot_result"]:
        plot_show_plotly(data.iloc[:,1:],anomaly_label,data["timestamps"],Data.id)

    return anomaly_label


def JsonAnomalyDetectionVibrationBoxplot(Data, k):
    # do something with Data
    # Data to pandas
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)

    # data timestamps 转换为日期格式
    datetime = pd.to_datetime(data.index, unit="ms")
    print(data.head(5))
    anomaly_label = []
    for column in data.columns:

        # 计算上下四分位数
        q1, q3 = np.percentile(data[column], [25, 75])
        iqr = q3 - q1

        # 计算离群值的范围
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr

        anomaly_label_i = []
        for value in data[column]:
            if (value < lower_bound).any() or (value > upper_bound).any():
                # print(value)
                anomaly_label_i.append(1)
            else:
                anomaly_label_i.append(0)
        anomaly_label.append(anomaly_label_i)

    # anomaly_label  转置
    anomaly_label_n = np.array(anomaly_label)
    anomaly_label_n_T = np.transpose(anomaly_label_n)
    anomaly_label_sum = [np.sum(x) for x in anomaly_label_n_T]

    # 绘制折线图，其中红色阴影表示异常
    if config["is_plot_result"]:
        plot_show_plotly(data,anomaly_label_sum,datetime,Data.id)

    return anomaly_label_sum



def JsonRepairTemperatureBoxplot(Data, k):
    # json to pandas
    # Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列
    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})
    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")
    # 计算上下四分位数
    q1, q3 = np.percentile(data['values'], [25, 75])
    iqr = q3 - q1

    # 计算离群值的范围
    upper_bound = q3 + k * iqr
    lower_bound = q1 - k * iqr

    # 标记异常值,如果数据小于下限或大于上限则为1，反之为0
    anomaly_label = []
    for value in data["values"]:
        if (value < lower_bound).any() or (value > upper_bound).any():
            # print(value)
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)


    # 修复
    data['repaired_values'] = data['values']
    # 遍历 anomaly_label,计算修复值
    for i in range(data.shape[0]):
        if anomaly_label[i] == 1:
            data.loc[i, 'repaired_values'] = upper_bound if data.loc[i, 'repaired_values'] > upper_bound else lower_bound

    # 绘图
    if config["is_plot_result"]:
        plot_show_single_repair(data['values'], data['repaired_values'], data["timestamps"], id=" ")


    return anomaly_label, data['repaired_values'].tolist()


def JsonRepairVibrationBoxplot(Data, k):
    # do something with data
    # Data to pandas
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)
    print(data)
    # data timestamps 转换为日期格式
    datetime = pd.to_datetime(data.index, unit="ms")

    # print(data)
    anomaly_label = []
    repairedValues = []
    for column in data.columns:

        # 计算上下四分位数
        q1, q3 = np.percentile(data[column], [25, 75])
        iqr = q3 - q1

        # 计算离群值的范围
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr

        anomaly_label_i = []
        repairedValue = []

        for value in data[column]:
            if (value < lower_bound).any() or (value > upper_bound).any():
                # 异常
                anomaly_label_i.append(1)
                # 修复
                repairedValue.append(upper_bound if value > upper_bound else lower_bound)
            else:
                # 没有异常
                anomaly_label_i.append(0)
                # 不用修复
                repairedValue.append(value)
        anomaly_label.append(anomaly_label_i)
        repairedValues.append(repairedValue)

    # anomaly_label  转置
    anomaly_label_n = np.array(anomaly_label)
    anomaly_label_n_T = np.transpose(anomaly_label_n)
    anomaly_label_sum = [np.sum(x) for x in anomaly_label_n_T]

    # repairedValues 转置
    repairedValues_n = np.array(repairedValues)
    repairedValues_n_T = np.transpose(repairedValues_n)
    repairedValues_T = repairedValues_n_T.tolist()

    # 绘图
    if config["is_plot_result"]:
        plot_show_Mult_repair(data.values, repairedValues_n,  datetime , Data.id)

    return anomaly_label_sum, repairedValues_T
