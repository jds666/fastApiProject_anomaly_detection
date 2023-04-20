# 导包
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from show.matplotlib_show import *
from show.plotly_show import *
import datetime
from scipy.stats import skew
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

plt.style.use('ggplot')


# 数据读取
def Read_Data():
    data0 = pd.read_csv('D:\大四材料\实习\梅钢\lz_eige_data\lz_eige_data.csv', header=None)  # 获取日期数据
    return data0


## 数据共 27736166 行
def Rename_Columns(data):
    ## 重命名字段
    data.columns = ['id', 'dev_no', 'dev_name', 'point_no', 'unit_no', 'ts', 'temperature', 'acceleration_peak_x',
                    'acceleration_peak_y',
                    'acceleration_peak_z', 'acceleration_rms_x', 'acceleration_rms_y', 'acceleration_rms_z',
                    'speed_peak_x', 'speed_peak_y',
                    'speed_peak_z', 'speed_rms_x', 'speed_rms_y', 'speed_rms_z', 'envelop_energy']


data = Read_Data()
Rename_Columns(data)

print("读入完成。。。")


def Data_Accuracy(data):
    ### 无效数据
    invalid_ratio = np.mean(data.isna())
    print("无效数据率:\n", invalid_ratio)
    O_invalid_ratio = mean(invalid_ratio)
    print("整体数据无效率: ", O_invalid_ratio)
    ### 重复数据
    duplicate_ratio = np.sum(data.duplicated())
    print("重复数据量:\n", duplicate_ratio)


## 缺失值检查
## 一定没有ts缺失，因为缺失已经被删了
# Data_Accuracy(data)

## 时间戳转换
# 去除所有ts是NAN的行
data.dropna(axis=0, subset=['ts'], inplace=True)

## 时间戳转为时间序列
data['ts'] = data['ts'].apply(lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
data['ts'] = pd.to_datetime(data['ts'])

print("时间戳转为时间序列完成。。。")

