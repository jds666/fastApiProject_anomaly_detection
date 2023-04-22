import json
import base64

import matplotlib.pyplot as plt
import uvicorn
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plot_data_from_csv import *
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from statsmodels.tsa.arima.model import ARIMA
import datetime
import os
import csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import load_model
from lstm.utils import *
from lstm.ploty_show import *


app = FastAPI()


with open('python_anomaly_detection.json') as file:
     config = json.load(file)

# 均值计算
def mean_fun(data):
    return np.mean(data)


class t_value(BaseModel):
    timestamp: int
    value: float


class Data(BaseModel):
    id: int

    devNo: str
    devName: str
    unitNo: str
    pointNo: str
    ts: int
    temperature: float

    accelerationPeakX: float
    accelerationRmsX: float
    speedPeakX: float
    speedRmsX: float

    accelerationPeakY: float
    accelerationRmsY: float
    speedPeakY: float
    speedRmsY: float

    accelerationPeakZ: float
    accelerationRmsZ: float
    speedPeakZ: float
    speedRmsZ: float

    envelopEnergy: float

class Data_repair():
    id: int

    devNo: str
    devName: str
    unitNo: str
    pointNo: str
    ts: int
    temperature: float

    accelerationPeakX: float
    accelerationRmsX: float
    speedPeakX: float
    speedRmsX: float

    accelerationPeakY: float
    accelerationRmsY: float
    speedPeakY: float
    speedRmsY: float

    accelerationPeakZ: float
    accelerationRmsZ: float
    speedPeakZ: float
    speedRmsZ: float

    envelopEnergy: float



class TemperatureInput(BaseModel):
    """
    temperature json input

    {
  "id": "2d1fe4feff8d79e0-S",
  "timestamps": [
    1675012379220,
    1675012679128,
    1675012979352,
    1675013279346,
    1675013579424,
    1675014179525,
    1675014179858,
    1675014479926,
    1675014779606,
    1675015079889,
    1675015379836,
    1675015679848
  ],
  "values": [
    7.625,
    7.625,
    7.5,
    7.375,
    7.375,
    7.3125,
    7.3125,
    7.375,
    7.4375,
    7.625,
    7.5625,
    7.75
  ]
}
    """
    id: str
    timestamps: List[int]
    values: List[float]

class TemperatureAnomalyOutput(BaseModel):
    id: str
    anomalyLabel: List[int]

class TemperatureRepairedOutput(BaseModel):
    id: str
    anomalyLabel: List[int]
    repairedValues: List[float]

'''
eigedata json input

{
    "id": 1,
    "devNo": 1,
    "devName": 1,
    "unitNo": 1,
    "pointNo": 1,
    "ts": 1,
    "temperature": 9999,
    "accelerationPeakX": 9999,
    "accelerationRmsX": 9999,
    "speedPeakX": 9999,
    "speedRmsX": 9999,
    "accelerationPeakY": 9999,
    "accelerationRmsY": 9999,
    "speedPeakY": 9999,
    "speedRmsY": 9999,
    "accelerationPeakZ": 9999,
    "accelerationRmsZ": 9999,
    "speedPeakZ": 9999,
    "speedRmsZ": 9999,
    "envelopEnergy": 9999
}

size: int
timeSeries: List[t_value]

{
  "size": 2,
  "timeSeries": [
    {
      "timestamp": 1001,
      "value": 1.1
    },
    {
      "timestamp": 1002,
      "value": 1.2
    }
  ]
}

'''


class VibrationInput(BaseModel):
    """
    vibration json input

    {
      "dimension": 12,
      "id": "2a1fe4feff8d79e0-Z",
      "timestamps": [
        1675012361452,
        1675012661446,
        1675012961519,
        1675013261599,
        1675013561706,
        1675013861819,
        1675014161827,
        1675014461942,
        1675014761971,
        1675015062071,
        1675015362095,
        1675015662133
      ],
      "valueNameList": [
        "AccelerationPeakX",
        "AccelerationRmsX",
        "SpeedPeakX",
        "SpeedRmsX",
        "AccelerationPeakY",
        "AccelerationRmsY",
        "SpeedPeakY",
        "SpeedRmsY",
        "AccelerationPeakZ",
        "AccelerationRmsZ",
        "SpeedPeakZ",
        "SpeedRmsZ"
      ],
      "values": [
        [
          3.159999847412109,
          0.6823999881744385,
          1.138000011444092,
          0.3981999754905701,
          5.589999675750732,
          1.284099936485291,
          1.639999985694885,
          0.5148000121116638,
          7.46999979019165,
          2.309999942779541,
          2.129999876022339,
          0.8267999887466431
        ],
        [
          6.819999694824219,
          1.369499921798706,
          2.319999933242798,
          0.674299955368042,
          7.37999963760376,
          1.769999980926514,
          2.009999990463257,
          0.6067999601364136,
          15.23999977111816,
          4.819999694824219,
          2.899999856948853,
          0.8495000004768372
        ],
        [
          9.6899995803833,
          2.710000038146973,
          2.589999914169312,
          0.9603999853134155,
          13.73999977111816,
          4.279999732971191,
          2.480000019073486,
          0.750499963760376,
          21.75,
          8.359999656677246,
          2.5,
          0.8246999979019165
        ],
        [
          10.25,
          2.679999828338623,
          3.069999933242798,
          0.9424999952316284,
          15.05999946594238,
          4.190000057220459,
          2.549999952316284,
          0.7392999529838562,
          19.10999870300293,
          6.329999923706055,
          2.369999885559082,
          0.7996999621391296
        ],
        [
          9.969999313354492,
          2.679999828338623,
          2.559999942779541,
          0.8868999481201172,
          14.88000011444092,
          4.480000019073486,
          2.240000009536743,
          0.7245999574661255,
          15.82999992370605,
          5.509999752044678,
          2.769999980926514,
          0.9490000009536743
        ],
        [
          10.30000019073486,
          2.669999837875366,
          2.75,
          0.9469999670982361,
          13.38999938964844,
          4.440000057220459,
          2.809999942779541,
          1.006899952888489,
          15.96999931335449,
          5.019999980926514,
          3.379999876022339,
          1.078400015830994
        ],
        [
          4.069999694824219,
          0.9509999752044678,
          1.32859992980957,
          0.4154999852180481,
          6.399999618530273,
          1.462599992752075,
          1.71999990940094,
          0.6014999747276306,
          6.539999961853027,
          2.029999971389771,
          1.899999976158142,
          0.7595999836921692
        ],
        [
          4.539999961853027,
          0.9442999958992004,
          1.361799955368042,
          0.4580999910831451,
          5.869999885559082,
          1.448199987411499,
          1.649999976158142,
          0.5532000064849854,
          6.739999771118164,
          2.059999942779541,
          2.240000009536743,
          0.7210999727249146
        ],
        [
          4.0,
          0.9357999563217163,
          1.336599946022034,
          0.4752999842166901,
          6.259999752044678,
          1.486699938774109,
          1.34850001335144,
          0.5138999819755554,
          7.480000019073486,
          2.099999904632568,
          2.259999990463257,
          0.8105999827384949
        ],
        [
          4.179999828338623,
          0.8996999859809875,
          1.362599968910217,
          0.4995999932289124,
          6.369999885559082,
          1.467299938201904,
          1.388499975204468,
          0.5346999764442444,
          6.170000076293945,
          2.089999914169312,
          2.5,
          0.8046999573707581
        ],
        [
          3.779999971389771,
          0.8382999897003174,
          1.2846999168396,
          0.3987999856472015,
          5.730000019073486,
          1.416399955749512,
          1.669999957084656,
          0.566100001335144,
          8.710000038146973,
          2.089999914169312,
          2.099999904632568,
          0.6311999559402466
        ],
        [
          7.25,
          1.459100008010864,
          2.220000028610229,
          0.7058999538421631,
          8.039999961853027,
          1.799999952316284,
          2.329999923706055,
          0.6743999719619751,
          14.67999935150146,
          5.230000019073486,
          2.899999856948853,
          0.9806999564170837
        ]
      ]
    }


    """
    id: str
    dimension: int
    timestamps: List[int]
    valueNameList: List[str]
    values: List[List[float]]

class VibrationAnomalyOutput(BaseModel):
    id: str
    dimension: int
    anomalyLabel: List[List[int]]

class VibrationRepairedOutput(BaseModel):
    id: str
    dimension: int
    anomalyLabel: List[List[int]]
    repairedValues: List[List[float]]

# 均值计算
@app.post('/anomaly_detection/threshold')
def anomaly_detection_threshold(data: Data):
    """
    K-sigma异常检测
    :param data:
    :return:
    """
    if data.temperature != 9999.0:
        ## 判断是否异常
        ## 从数据库中读取阈值

        return {"measured_temperature": data.temperature}
    elif data.speedRmsY != 9999.0:
        return {"speed_rms_y": data.speedRmsY}
    elif data.speedRmsX != 9999.0:
        return {"speed_rms_x": data.speedRmsX}
    else:
        return {"message": "No data"}

##########################################################################################
########################################## Done ##########################################
##########################################################################################

@app.get('/anomaly_detection/temperature/k_sigma')
def anomaly_detection_temperature_k_sigma(path: str,k = config["k"]):
    """
    单维异常检测 k-sigma
    :param path:
    :param k:
    :return:
    """
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/temperature/raw'

    ################################################
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            # sum_data = 0
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                # 读取文件到pandas
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',', names=['timestamp', 'temperature'])

                    # 时间戳转换成日期
                    data['timestamp'] = data['timestamp'].apply(
                        lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                    data['timestamp'] = pd.to_datetime(data['timestamp'])

                    # 找出所有重复行
                    same_timestamp_rows = data[data.duplicated(keep='first')]

                    # data去除重复值,只保留最后一个
                    data = data.drop_duplicates(subset='timestamp', keep='last')

                    # 计算平均值和标准差
                    mean = data.temperature.mean()
                    std = data.temperature.std()
                    # 计算异常值
                    outliers = data[((data.temperature - mean) / std) > k]
                    # 保存结果到文件中
                    output_path = subdir_path.replace("raw\\", "output/")
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)

                    same_timestamp_rows_filename = os.path.join(output_path, 'duplicated.csv')
                    outliers_filename = os.path.join(output_path, 'outliers.csv')

                    same_timestamp_rows.to_csv(same_timestamp_rows_filename, index=False)
                    outliers.to_csv(outliers_filename, index=False)


    ################################################
    # return {"原数据行数": data.shape[0],
    #         "重复点数量：": same_timestamp_rows.shape[0],
    #         "清洗之后数据行数": data.shape[0] - same_timestamp_rows.shape[0],
    #         "异常点数量：": outliers.shape[0]}
    return {"message:": "success"}


@app.get('/repair/temperature/k_sigma')
def repair_temperature_k_sigma(path: str,k = config["k"]):
    """
    单维异常检测 k-sigma
    :param path:
    :param k:
    :return:
    """
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/temperature/raw'


    ################################################
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            # sum_data = 0
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                # 读取文件到pandas
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',', names=['timestamp', 'temperature'])

                    # 时间戳转换成日期
                    data['timestamp'] = data['timestamp'].apply(
                        lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                    data['timestamp'] = pd.to_datetime(data['timestamp'])

                    # 比较相邻行中的时间戳和温度，如果相同则为1，否则为0
                    data['same_label'] = (data.duplicated(keep='first')).astype(int)
                    #
                    # # 将第一行的'label'设为0，因为第一行没有前一行可以比较
                    data.loc[data.index[0], 'same_label'] = 0

                    # data去除重复值,只保留最后一个
                    # data = data.drop_duplicates(subset='timestamp', keep='last')
                    # 输出重复行数
                    print(data.duplicated().sum())

                    # 计算平均值和标准差
                    mean = data.temperature.mean()
                    std = data.temperature.std()

                    # 计算异常值
                    outliers = data[((data.temperature - mean) / std) > k]
                    data['outliers_label'] = (((data.temperature - mean) / std) > k).astype(int)

                    #                     if abs(outliers - mean) > k * std:
                    #                         fixed_a[i + j] = mean + k * std if outliers > mean else mean - k * std
                    # 修复,异常数据就修复为 k-sigma 阈值

                    data['repaired_value'] = data['temperature']
                    # 遍历outliers_label,计算修复值
                    for i in range(data.shape[0]):
                        if data.loc[i,'outliers_label'] == 1:
                            data.loc[i,'repaired_value'] = mean + k * std if data.loc[i,'repaired_value'] > mean else mean - k * std


                    print(data.head())

                    # 修复结果保存到文件中
                    output_path = subdir_path.replace("raw\\", "output/")
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)

                    # 文件名
                    repaired_filename = os.path.join(output_path, 'repaired.csv')

                    data.to_csv(repaired_filename, index=False)


    ################################################
    # return {"原数据行数": data.shape[0],
    #         "重复点数量：": same_timestamp_rows.shape[0],
    #         "清洗之后数据行数": data.shape[0] - same_timestamp_rows.shape[0],
    #         "异常点数量：": outliers.shape[0]}
    return {"message:": "success"}

@app.get('/anomaly_detection/vibration/k_sigma')
def anomaly_detection_vibration_k_sigma(path: str,k = config["k"], halfdaynum = 144):
    """
    高维异常检测 k-sigma
    :param path:
    :param k:
    :param halfdaynum:
    :return:
    """
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/vibration/raw'
    ################################################
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            # sum_data = 0
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)

                # 记录索引
                same_rows_arrays = []
                outliers_arrays = []

                # 读取文件到pandas
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',')

                    # name for data,first index is timestamp,others are value
                    if data.shape[1] == 5:
                        data.columns = ['timestamp', 'AccelerationPeakX', 'AccelerationRmsX', 'SpeedPeakX', 'SpeedRmsX']
                    else:
                        data.columns = ['timestamp','AccelerationPeakX', 'AccelerationRmsX', 'SpeedPeakX', 'SpeedRmsX',
                                        'AccelerationPeakY', 'AccelerationRmsY', 'SpeedPeakY', 'SpeedRmsY',
                                        'AccelerationPeakZ', 'AccelerationRmsZ', 'SpeedPeakZ', 'SpeedRmsZ']
                    # 时间戳转换成日期
                    data['timestamp'] = data['timestamp'].apply(
                        lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                    data['timestamp'] = pd.to_datetime(data['timestamp'])

                    # 分列找出每列的重复值、异常值，存入各自duplicated.csv
                    for col in data.columns:
                        if col == 'timestamp':
                            continue
                        col_data = data[['timestamp', col]].copy()
                        # 找出col_data中timestamp、col都重复的行
                        same_rows = col_data[col_data.duplicated(subset={'timestamp', col}, keep=False) & (
                                    col_data['timestamp'] == col_data['timestamp'].shift(1))]

                        # data去除重复值,只保留最后一个
                        col_data = col_data.drop_duplicates(subset='timestamp', keep='last')

                        # 计算平均值和标准差
                        mean = col_data[col].mean()
                        std = col_data[col].std()

                        # 计算异常值
                        outliers = col_data[((col_data[col] - mean) / std) > k]

                        # save to csv
                        output_path = subdir_path.replace("raw\\", "output/")
                        if not os.path.exists(output_path):
                            os.mkdir(output_path)

                        same_rows_filename = os.path.join(output_path, col+'-duplicated.csv')
                        same_rows.to_csv(same_rows_filename, index=False)

                        outliers_filename = os.path.join(output_path,col+'-outliers.csv')
                        outliers.to_csv(outliers_filename, index=False)

                        # 记录索引
                        same_rows_index = same_rows.index.tolist()
                        same_rows_arrays.append(same_rows_index[:min(halfdaynum, len(same_rows_index))])
                        outliers_index = outliers.index.tolist()
                        outliers_arrays.append(outliers_index[:min(halfdaynum, len(outliers_arrays))])

                    ## 都重复、异常的点保存到duplicated.csv、outliers.csv
                    # outliers_arrays列表中有 len(outliers_arrays) 个单维递增列表数据，所有数如果在任意 len(outliers_arrays)//3 个及以上的单维递增列表中出现，则将该数记录为重点异常，记录到 same_rows_vital 列表中
                    outliers_vital = []
                    # 创建 一个 字典来统计每个数在哪些单维递增列表中出现
                    num_count = {}
                    for arr in outliers_arrays:
                        for num in arr:
                            if num not in num_count:
                                num_count[num] = set()
                            num_count[num].add(outliers_arrays.index(arr))

                    # 判断哪些数出现在了3个及以上的单维递增列表中
                    for num, count in num_count.items():
                        if len(count) >=  len(outliers_arrays)//3:
                            outliers_vital.append(num)

                    # save to csv
                    outliers_filename =os.path.join(subdir_path.replace("raw\\", "output/"), 'outliers.csv')
                    data.iloc[outliers_vital,:].to_csv(outliers_filename, index=False)


    ################################################

    return {"message:": "success"}


@app.get('/repair/vibration/k_sigma')
def repair_vibration_k_sigma(path: str,k = config["k"], halfdaynum = 144):
    ################################################
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            # sum_data = 0
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)

                # 读取文件到pandas
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',', nrows =halfdaynum)
                    # name for data,first index is timestamp,others are value
                    if data.shape[1] == 5:
                        data.columns = ['timestamp', 'AccelerationPeakX', 'AccelerationRmsX', 'SpeedPeakX', 'SpeedRmsX']
                    else:
                        data.columns = ['timestamp', 'AccelerationPeakX', 'AccelerationRmsX', 'SpeedPeakX', 'SpeedRmsX',
                                        'AccelerationPeakY', 'AccelerationRmsY', 'SpeedPeakY', 'SpeedRmsY',
                                        'AccelerationPeakZ', 'AccelerationRmsZ', 'SpeedPeakZ', 'SpeedRmsZ']

                    # 时间戳转换成日期
                    data['timestamp'] = data['timestamp'].apply(
                        lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                    data['timestamp'] = pd.to_datetime(data['timestamp'])

                    anomaly_label= []
                    repairedValues = []
                    # 分列找出每列的重复值、异常值，存入各自duplicated.csv
                    for col in data.columns:
                        # 计算平均值和标准差
                        mean = data[col].mean()
                        std = data[col].std()

                        # 标记异常值,如果(data.values - mean) / std > k为1，反之为0
                        anomaly_label_i = []
                        repairedValue = []

                        for value in data[col]:
                            if (value - mean) / std > k:
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

                    # repairedValues 转置
                    repairedValues_n = np.array(repairedValues)
                    repairedValues_n_T = np.transpose(repairedValues_n)
                    repairedValues_T = repairedValues_n_T.tolist()

                    # 将行和列放到数据集中
                    # data_repaired = pd.DataFrame(np.array(repairedValues_T), index=repairedValues_T[:][1])
                    # data_repaired['anomaly_label'] = np.array(anomaly_label)
                    data.values[:,:] = np.array(repairedValues_T)
                    # 存入csv
                    output_path = subdir_path.replace("raw\\", "output/")
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)

                    repaired_filename = os.path.join(output_path, 'reparied')
                    data.to_csv(repaired_filename, index=False)


    ################################################

    return {"message:": "success"}



@app.post("/Json/anomaly_detection/temperature/k-sigma")
async def Json_anomaly_detection_temperature_k_sigma(Data: TemperatureInput, k = config["k"]):
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

    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "anomalyLabel": anomaly_label}
    response = TemperatureAnomalyOutput(**response_data)

    ################################################
    return response.dict()



@app.post("/Json/anomaly_detection/vibration/k-sigma")
async def Json_anomaly_detection_vibration_k_sigma(Data: VibrationInput,k = 2):
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
    anomaly_label_T = anomaly_label_n_T.tolist()

    # 将标记数据转换回json格式
    response_data = {"id": Data.id,"dimension": Data.dimension,"anomalyLabel": anomaly_label_T}
    response = VibrationAnomalyOutput(**response_data)
    ################################################
    return response.dict()






@app.post("/Json/repair/temperature/k-sigma")
async def Json_repair_temperature_k_sigma(Data: TemperatureInput,k= config["k"]):
    # json to pandas
    # Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列
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


    # 修复
    data['repaired_values'] = data['values']
    # 遍历 anomaly_label,计算修复值
    for i in range(data.shape[0]):
        if anomaly_label[i] == 1:
            data.loc[i, 'repaired_values'] = mean + k * std if data.loc[i, 'repaired_values'] > mean else mean - k * std

    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "anomalyLabel": anomaly_label,"repairedValues": data['repaired_values'].tolist()}
    response = TemperatureRepairedOutput(**response_data)
    ################################################
    return response.dict()





@app.post("/Json/repair/vibration/k-sigma")
async def Json_repair_vibration_k_sigma(Data: VibrationInput,k = config["k"]):
    # do something with data
    # Data to pandas
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)

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
    anomaly_label_T = anomaly_label_n_T.tolist()

    # repairedValues 转置
    repairedValues_n = np.array(repairedValues)
    repairedValues_n_T = np.transpose(repairedValues_n)
    repairedValues_T = repairedValues_n_T.tolist()

    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "dimension": Data.dimension, "anomalyLabel": anomaly_label_T,"repairedValues":repairedValues_T}
    response = VibrationRepairedOutput(**response_data)
    ################################################

    return response.dict()




@app.post("/Json/anomaly_detection/temperature/boxplot")
async def Json_anomaly_detection_temperature_boxplot(Data: TemperatureInput, k = config["k"]):
    # json to pandas
    #Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列
    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})

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
            print(value)
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)

    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "anomalyLabel": anomaly_label}
    response = TemperatureAnomalyOutput(**response_data)

    ################################################
    return response.dict()



@app.post("/Json/anomaly_detection/vibration/boxplot")
async def Json_anomaly_detection_vibration_boxplot(Data: VibrationInput,k = config["k"]):
    # do something with Data
    # Data to pandas
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)

    # print(data)
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
                print(value)
                anomaly_label_i.append(1)
            else:
                anomaly_label_i.append(0)
        anomaly_label.append(anomaly_label_i)

    # anomaly_label  转置
    anomaly_label_n = np.array(anomaly_label)
    anomaly_label_n_T = np.transpose(anomaly_label_n)
    anomaly_label_T = anomaly_label_n_T.tolist()

    # 将标记数据转换回json格式
    response_data = {"id": Data.id,"dimension": Data.dimension,"anomalyLabel": anomaly_label_T}
    response = VibrationAnomalyOutput(**response_data)
    ################################################
    return response.dict()






@app.post("/Json/repair/temperature/boxplot")
async def Json_repair_temperature_boxplot(Data: TemperatureInput,k= config["k"]):
    # json to pandas
    # Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列
    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})

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
            print(value)
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)


    # 修复
    data['repaired_values'] = data['values']
    # 遍历 anomaly_label,计算修复值
    for i in range(data.shape[0]):
        if anomaly_label[i] == 1:
            data.loc[i, 'repaired_values'] = upper_bound if data.loc[i, 'repaired_values'] > upper_bound else lower_bound

    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "anomalyLabel": anomaly_label,"repairedValues": data['repaired_values'].tolist()}
    response = TemperatureRepairedOutput(**response_data)
    ################################################
    return response.dict()





@app.post("/Json/repair/vibration/boxplot")
async def Json_repair_vibration_boxplot(Data: VibrationInput,k = config["k"]):
    # do something with data
    # Data to pandas
    data = pd.DataFrame(Data.values, columns=Data.valueNameList, index=Data.timestamps)

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
    anomaly_label_T = anomaly_label_n_T.tolist()

    # repairedValues 转置
    repairedValues_n = np.array(repairedValues)
    repairedValues_n_T = np.transpose(repairedValues_n)
    repairedValues_T = repairedValues_n_T.tolist()

    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "dimension": Data.dimension, "anomalyLabel": anomaly_label_T,"repairedValues":repairedValues_T}
    response = VibrationRepairedOutput(**response_data)
    ################################################

    return response.dict()

##########################################################################################
########################################## Test ##########################################
##########################################################################################

@app.get('/anomaly_detection/temperature/knn')
def anomaly_detection_temperature_knn(path: str):
    """
    单维异常检测 k-sigma
    :param path:
    :param k:
    :return:
    """
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/temperature/raw'


    ################################################
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            # sum_data = 0
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                # 读取文件到pandas
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',', names=['timestamp', 'temperature'])

                    # 时间戳转换成 unix
                    timestamps = data['timestamp'].values
                    unix_timestamps = np.array(
                        [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() for ts in timestamps])

                    # 找出所有重复行
                    same_timestamp_rows = data[data.duplicated(subset='timestamp', keep=False) & (data['timestamp'] == data['timestamp'].shift(1))]

                    # data去除重复值,只保留最后一个
                    data = data.drop_duplicates(subset='timestamp', keep='last')

                    # 计算lof
                    #####################################################################

                    # 将数据转换为 PyTorch 张量
                    temperatures = data['temperature'].values
                    X = torch.tensor(np.column_stack((unix_timestamps, temperatures)), dtype=torch.float32)


                    # 计算最近邻居
                    k = 5
                    distances, indices = knn(X, X, k)

                    # 计算 LOF 得分
                    lof = torch.zeros(X.shape[0])
                    for i in range(X.shape[0]):
                        k_distances = distances[i][1:]
                        k_indices = indices[i][1:]
                        reach_distances = torch.max(torch.stack([k_distances, distances[k_indices, 1]], dim=1), dim=1)[
                            0]
                        lof[i] = torch.mean(reach_distances) / k_distances.mean()

                    # 打印异常点
                    threshold = 1.5
                    anomalies_idx = torch.where(lof > threshold)[0].tolist()
                    outliers = data.iloc[anomalies_idx, :]
                    print(outliers)

                    #####################################################################

                    # 保存结果到文件中
                    output_path = subdir_path.replace("raw\\", "output/")
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)

                    same_timestamp_rows_filename = os.path.join(output_path, 'duplicated.csv')
                    outliers_filename = os.path.join(output_path, 'outliers.csv')

                    same_timestamp_rows.to_csv(same_timestamp_rows_filename, index=False)
                    outliers.to_csv(outliers_filename, index=False)


    ################################################
    # return {"原数据行数": data.shape[0],
    #         "重复点数量：": same_timestamp_rows.shape[0],
    #         "清洗之后数据行数": data.shape[0] - same_timestamp_rows.shape[0],
    #         "异常点数量：": outliers.shape[0]}
    return {"message:": "success"}

@app.post("/Json/anomaly_detection/temperature/lstm")
async def Json_anomaly_detection_temperature_lstm(Data: TemperatureInput):

    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})
    # data timestamps 转换为日期格式
    data["timestamps"] = pd.to_datetime(data.timestamps, unit="ms")

    data.set_index("timestamps", inplace=True)

    model_path = r'D:\Pyprogram\fastApiProject_anomaly_detection\lstm\model\temperature\0a1ee4feff8d79e0-S_lstm_model.h5'
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
    anomalies_data = scaler.inverse_transform(test_data[anomalies[0], :])

    # 绘制原始数据和异常点
    plot_show_plotly(original_data, anomalies_data, anomalies, 0)


    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "anomalyLabel": anomaly_label}
    response = TemperatureAnomalyOutput(**response_data)

    ################################################
    return response.dict()
    # return {"message:","succeeded"}


@app.post("/Json/repair/temperature/arma")
async def Json_repair_temperature_arma(Data: TemperatureInput, k= config["k"]):

    # json to pandas
    # Data 中的timestamps和values,读取到pandas中成为“timestamps”，“values”两列
    data = pd.DataFrame({"timestamps": Data.timestamps, "values": Data.values})

    # Fit an ARIMA model to the data
    model = ARIMA(data['values'], order=(1, 1, 0)).fit()
    # print(model.summary())
    print(model.forecast())
    # Use the model to predict the next value
    predicted_value = model.forecast()[0][0]


    # Compare the predicted value with the actual value, and mark as anomaly if difference is greater than threshold
    anomaly_label = []
    threshold = k * model.resid.std()
    for i in range(len(data.values)):
        if abs(data.loc[i, "values"] - predicted_value) > threshold:
            anomaly_label.append(1)
        else:
            anomaly_label.append(0)

    # Repair the data using the predicted value
    data['repaired_values'] = data['values']
    data.loc[anomaly_label == 1, 'repaired_values'] = predicted_value

    # 将标记数据转换回json格式
    response_data = {"id": Data.id, "anomalyLabel": anomaly_label,
                     "repairedValues": data['repaired_values'].tolist()}
    response = TemperatureRepairedOutput(**response_data)
    ################################################
    return response.dict()


# @app.route("/status")
# def get_status():
#     return '<h1> Hello, World! </h1>'
def generate_html_response():
    html_content = """
    <!DOCTYPE html>
<html>
  <head>
    <title>泰山介绍</title>
  </head>
  <body>
    <h1>泰山</h1> 
    <p>泰山是中国五岳之首，位于山东省泰安市境内，海拔1545米。泰山是道教和儒家文化的重要象征，有着悠久的历史和文化底蕴。泰山有五岳之冠、天下第一山的美誉，是中国著名的旅游胜地。</p>
    <h2>泰山的特点</h2>
    <ul>
      <li>巍峨壮观：泰山主峰玉皇顶海拔1545米，高耸入云。</li>
      <li>险峻奇特：泰山山势陡峭，有七十二峰、九十九壑、一百八十石阶。</li>
      <li>历史文化：泰山有着丰富的历史和文化遗产，是道教和儒家文化的重要象征。</li>
      <li>旅游胜地：泰山是中国著名的旅游胜地，吸引着众多国内外游客前来观光、登山。</li>
    </ul>
  </body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/plot", response_class=HTMLResponse)
async def Json_anomaly_detection_temperature_k_sigma_plot(Data: TemperatureInput, k=3):
    timestamps = Data.timestamps
    values = Data.values

    # Calculate the mean and standard deviation of the values
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    # Identify anomalies (values more than k standard deviations from the mean)
    anomalies = [i for i in range(len(values)) if abs(values[i] - mean) > k * std_dev]

    # Create a line plot of the timestamps and values
    fig, ax = plt.subplots()
    ax.plot(timestamps, values)
    ax.set(xlabel='Timestamps', ylabel='Temperature (Celsius)',
           title='Temperature vs Time')
    ax.grid()

    # Add red dots to the plot to highlight the anomalies
    ax.scatter([timestamps[i] for i in anomalies], [values[i] for i in anomalies], color='red')

    # Save the plot to a PNG file
    fig.savefig("show/png/temperature_plot.png")

    # Show the plot in a separate window (this is optional)
    #plt.show()

    # Return the HTML content to display the plot in the frontend
    with open("show/png/temperature_plot.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    html_content = f'<img src="data:image/png;base64,{encoded_image}">'
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/items/", response_class=HTMLResponse)
async def read_items():
    return generate_html_response()

dash_app = create_dash_app(requests_pathname_prefix="/dash/")
app.mount("/dash", WSGIMiddleware(dash_app.server))

# 主函数
if __name__ == "__main__":
    uvicorn.run(app='main:app', host='127.0.0.1', port=8181, reload=False)
