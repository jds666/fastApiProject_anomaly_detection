import pandas as pd
import numpy as np
from lstm.utils import *
from show.plotly_show import *

config = get_config_from_json(json_file)

def JsonAnomalyDetectionTemperatureKSigma(Data, k):
    # json to pandas
    #Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列

    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})

    # 计算平均值和标准差
    mean = data['values'].mean()
    std = data['values'].std()

    # 标记异常值,如果(data.values - mean) / std > k为1，反之为0
    anomaly_label = []
    for i in range(len(data.values)):
        if abs(data.loc[i, "values"] - mean) / std > k:
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)

    return anomaly_label



def JsonAnomalyDetectionVibrationKSigma(Data,k):
    # do something with Data
    # Data to pandas
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)


    # print(data)
    anomaly_label = []
    for column in data.columns:
        # 计算平均值和标准差
        mean = data[column].mean()
        std = data[column].std()


        # print("==============================================")
        # print("mean: ", mean)
        # print("std: ", std)

        # 标记异常值,如果(data.values - mean) / std > k为1，反之为0
        anomaly_label_i = []
        for value in data[column]:
            # print("value============>>", value)
            if abs(value - mean) / std > k:
                anomaly_label_i.append(1)
            else:
                anomaly_label_i.append(0)
        anomaly_label.append(anomaly_label_i)

    # anomaly_label  转置
    anomaly_label_n = np.array(anomaly_label)
    anomaly_label_n_T = np.transpose(anomaly_label_n)
    anomaly_label_sum = [np.sum(x) for x in anomaly_label_n_T]

    return anomaly_label_sum

def JsonRepairTemperatureKSigma(Data,k):
    # json to pandas
    # Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列
    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})
    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")
    # 计算平均值和标准差
    mean = data['values'].mean()
    std = data['values'].std()

    # 标记异常值,如果(data.values - mean) / std > k为1，反之为0
    anomaly_label = []
    for i in range(len(data.values)):
        if abs(data.loc[i, "values"] - mean) / std > k:
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)

    # 修复
    data['repaired_values'] = data['values']
    # 遍历 anomaly_label,计算修复值
    for i in range(data.shape[0]):
        if anomaly_label[i] == 1:
            data.loc[i, 'repaired_values'] = mean + k * std if data.loc[i, 'repaired_values'] > mean else mean - k * std
    print(data['repaired_values'].tolist())

    # 绘图
    if config["is_plot_result"]:
        plot_show_single_repair(data['values'], data['repaired_values'], data["timestamps"], Data.id)

    return anomaly_label, data['repaired_values'].tolist()


def JsonRepairVibrationKSigma(Data, k):
    # do something with data
    # Data to pandas
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)

    # data timestamps 转换为日期格式
    data.timestamps = pd.to_datetime(data.index, unit="ms")

    # print(data)
    anomaly_label = []
    repairedValues = []
    for column in data.columns:
        # 计算平均值和标准差
        mean = data[column].mean()
        std = data[column].std()

        # 标记异常值,如果(data.values - mean) / std > k为1，反之为0
        anomaly_label_i = []
        repairedValue = []
        for value in data[column]:
            if abs(value - mean) / std > k:
                # 异常
                anomaly_label_i.append(1)
                # 修复
                repairedValue.append(mean + k * std if value > mean else mean - k * std)
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
        plot_show_Mult_repair(data.values, repairedValues_n,  data.timestamps , Data.id)

    return anomaly_label_sum, repairedValues_T
