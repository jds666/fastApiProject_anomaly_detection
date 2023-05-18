import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

from utils import *

# 获取配置文件
config = get_config_from_json(json_file)

def lstm_model(data, config, model_path):
    # 转换数据格式
    scaler = MinMaxScaler()  # 归一化
    data = scaler.fit_transform(data)
    # 将数据拆分为训练集和测试集
    train_size = int(len(data) * 0.9)
    train_data = data[500:train_size, :]
    test_data = data[train_size:, :]

    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(train_data.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练LSTM模型
    model.fit(train_data[:, :, np.newaxis], train_data[:, 0], epochs=config["lstm_epochs"],
              batch_size=config["lstm_batch_size"])
    model.save(model_path)




# 训练整个文件夹
def train(k):
    for root, dirs, files in os.walk(train_datasets_path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            i = 0
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                print(i,":",file_path)
                # 获得保存模型与html的路径
                model_path = create_model_paths(file_path)
                data = pd.read_csv(file_path, index_col=0)
                lstm_model(data, config,model_path)
                i = i+1
                print("iiiiii=====================:  ",i)
                if i >= k:
                    break

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(tf.config.list_physical_devices('GPU'))
configG = tf.compat.v1.ConfigProto()
configG.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configG)
tf.compat.v1.keras.backend.set_session(session)

# 训练整批
# train(1)

# 指定某一个训练
file_path = r'D:\Pyprogram\Python_Data_Analysis\data_csv\wired_data\wired_point_no_50294D201003011.csv'
model_path = create_model_paths(file_path)
data = pd.read_csv(file_path, index_col=0)
lstm_model(data, config,model_path)



