import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import load_model
from lstm.utils import *
from lstm.ploty_show import *

config = get_config_from_json(json_file)


def JsonAnomalyDetectionTemperatureLstm(Data):

    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})
    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")
    # data['timestamps'] = data['timestamps'].apply(
    #     lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
    # data['timestamps'] = pd.to_datetime(data['timestamps'])

    data.set_index("timestamps", inplace=True)
    data_time =[str(ts) for ts in data.index.tolist()]

    model_path = config["model_path"]+'\\model\\temperature\\'+Data.id+'_lstm_model.h5'
    loaded_model = load_model(model_path)

    ########################推理##############################
    scaler = MinMaxScaler()  # 归一化
    data = scaler.fit_transform(data)
    train_size = 0
    test_data = data[train_size:, :]
    #########################################################

    # 对测试集进行预测
    test_predict = loaded_model.predict(test_data[:, :, np.newaxis])
    test_predict = np.squeeze(test_predict)

    # 计算平均绝对误差
    mae = np.mean(np.abs(test_predict - test_data[:, 0]))

    # 根据平均绝对误差确定异常点
    threshold = mae * config["lstm_threshold_mea_k"]
    anomalies = np.where(np.abs(test_predict - test_data[:, 0]) > threshold)
    # print(f"异常点：{anomalies[0]}")

    # 标记异常值,如果在anomalies[0]中为1，反之为0
    anomaly_label = []
    for i in range(len(Data.values)):
        if i in anomalies[0]:
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)

    # 反归一化数据
    original_data = scaler.inverse_transform(data)

    # 绘制原始数据和异常点
    if config["is_plot_result"]:
        plot_show_plotly(original_data, anomalies, data_time,["temperature"], 0, id = Data.id)

    return anomaly_label



def JsonRepairTemperatureLstm(Data):

    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})
    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")

    data.set_index("timestamps", inplace=True)
    data_time = [str(ts) for ts in data.index.tolist()]

    model_path = config["model_path"]+'\\model\\temperature\\'+Data.id+'_lstm_model.h5'
    loaded_model = load_model(model_path)

    ########################推理##############################
    scaler = MinMaxScaler()  # 归一化
    data = scaler.fit_transform(data)
    train_size = 0
    test_data = data[train_size:, :]
    #########################################################

    # 对测试集进行预测
    test_predict = loaded_model.predict(test_data[:, :, np.newaxis])
    repair_data = test_predict
    test_predict = np.squeeze(test_predict)

    # 计算平均绝对误差
    mae = np.mean(np.abs(test_predict - test_data[:, 0]))

    # 根据平均绝对误差确定异常点
    threshold = mae * config["lstm_threshold_mea_k"]
    anomalies = np.where(np.abs(test_predict - test_data[:, 0]) > threshold)

    # print(threshold)
    # print(f"异常点：{anomalies[0]}")

    # 标记异常值,如果在anomalies[0]中为1，反之为0
    anomaly_label = []
    for i in range(len(Data.values)):
        if i in anomalies[0]:
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)

    # 反归一化数据
    original_data = scaler.inverse_transform(data)
    repair_data = scaler.inverse_transform(repair_data)

    #[:5]) 绘制原始数据和异常点
    # plot_show_plotly(original_data, anomalies, 0,id = Data.id)
    #plot_show_plotly_repair(original_data, repair_data, data_time,id=Data.id)
    if config["is_plot_result"]:

        plot_show_plotly_repair(original_data, repair_data, data_time,["temperature"], id=Data.id)
    repair_data = np.squeeze(repair_data).tolist()
    return anomaly_label, repair_data



def JsonAnomalyDetectionVibrationLstm(Data):

    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)
    # data timestamps 转换为日期格式
    data.index = pd.to_datetime(data.index.values, unit="ms")

    data_time = [str(ts) for ts in data.index.tolist()]

    # 加载模型
    if Data.dimension == 4:
        model_path = config["model_path"]+'\\model\\wired_data\\'+Data.id+'_lstm_model.h5'
    elif Data.dimension == 12:
        model_path = config["model_path"]+'\\model\\wireless_data\\'+Data.id+'_lstm_model.h5'
    else:
        model_path = config["model_path"]+'\\model\\temperature\\'+Data.id+'_lstm_model.h5'

    loaded_model = load_model(model_path)


    ########################推理##############################
    scaler = MinMaxScaler()  # 归一化
    # print(data.head())
    data = scaler.fit_transform(data)
    train_size = 0
    test_data = data[train_size:, :]
    #########################################################

    # 对测试集进行预测
    # print(test_data[:, :, np.newaxis])
    test_predict = loaded_model.predict(test_data[:, :, np.newaxis])
    test_predict = np.squeeze(test_predict)


    mae = np.mean(np.abs(test_predict[:, 0] - test_data[:, 0]))
    # 根据平均绝对误差确定异常点
    threshold = mae * config["lstm_threshold_mea_k"]
    anomalies_max = np.where(np.abs(test_predict[:,0] - test_data[:, 0]) > threshold)

    anomaly_label_all = []
    for j in range(Data.dimension):
        # 计算平均绝对误差
        mae = np.mean(np.abs(test_predict[:,j] - test_data[:, j]))
        # 根据平均绝对误差确定异常点
        threshold = mae * config["lstm_threshold_mea_k"]
        anomalies = np.where(np.abs(test_predict[:,j] - test_data[:, j]) > threshold)
        if np.array(anomalies_max).shape[1] < np.array(anomalies).shape[1]:
            anomalies_max = anomalies
        anomaly_label = []
        for i in range(len(Data.values)):
            if i in anomalies[0]:
                anomaly_label.append(1)
            else:
                anomaly_label.append(0)
        anomaly_label_all.append(anomaly_label)

    # anomaly_label_all  转置
    anomaly_label_all_n = np.array(anomaly_label_all)
    anomaly_label_all_n_T = np.transpose(anomaly_label_all_n)
    anomaly_label_sum = [ np.sum(x) for x in anomaly_label_all_n_T]


    # 反归一化数据
    original_data = scaler.inverse_transform(data)
    if config["is_plot_result"]:
        plot_show_plotly(original_data, anomalies_max,data_time,Data.valueNameList, 0,id = Data.id)

    return anomaly_label_sum



def JsonRepairVibrationLstm(Data):

    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)
    # data timestamps 转换为日期格式
    data.index = pd.to_datetime(data.index.values, unit="ms")

    data_time = [str(ts) for ts in data.index.tolist()]

    # 加载模型
    if Data.dimension == 4:
        model_path = config["model_path"]+'\\model\\wired_data\\'+Data.id+'_lstm_model.h5'
    elif Data.dimension == 12:
        model_path = config["model_path"]+'\\model\\wireless_data\\'+Data.id+'_lstm_model.h5'
    else:
        model_path = config["model_path"]+'\\model\\temperature\\'+Data.id+'_lstm_model.h5'

    loaded_model = load_model(model_path)

    ########################推理##############################
    scaler = MinMaxScaler()  # 归一化
    data = scaler.fit_transform(data)
    train_size = 0
    test_data = data[train_size:, :]
    #########################################################

    # 对测试集进行预测
    test_predict = loaded_model.predict(test_data[:, :, np.newaxis])
    repair_data = test_predict
    test_predict = np.squeeze(test_predict)

    # 反归一化数据
    original_data = scaler.inverse_transform(data)
    repair_data = scaler.inverse_transform(repair_data)

    repairedValues = original_data.tolist()

    anomaly_label_all = []
    for j in range(Data.dimension):
        # 计算平均绝对误差
        mae = np.mean(np.abs(test_predict[:,j] - test_data[:, j]))
        # 根据平均绝对误差确定异常点
        threshold = mae * config["lstm_threshold_mea_k"]
        anomalies = np.where(np.abs(test_predict[:,j] - test_data[:, j]) > threshold)
        # print(j,f"异常点：{anomalies[0]}")

        # 标记异常值,如果在anomalies[0]中为1，反之为0
        anomaly_label = []
        for i in range(len(Data.values)):
            if i in anomalies[0]:
                anomaly_label.append(1)
                repairedValues[i][j] = repair_data[i][j]
            else:
                anomaly_label.append(0)
        anomaly_label_all.append(anomaly_label)

    # anomaly_label_all  转置
    anomaly_label_all_n = np.array(anomaly_label_all)
    anomaly_label_all_n_T = np.transpose(anomaly_label_all_n)
    anomaly_label_sum = [ np.sum(x) for x in anomaly_label_all_n_T]

    # 绘制原始数据和预测数据
    if config["is_plot_result"]:
        plot_show_plotly_repair(original_data, repair_data, data_time,Data.valueNameList, id=Data.id)

    return anomaly_label_sum, repairedValues
