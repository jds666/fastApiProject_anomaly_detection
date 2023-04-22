import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from utils import *
from ploty_show import *


json_file = r"D:\Pyprogram\fastApiProject_anomaly_detection\python_anomaly_detection.json"
data_file = r"D:\Pyprogram\Python_Data_Analysis\data_csv\temperature\Temperature_point_no_0a1ee4feff8d79e0-S.csv"
# data_file = r"D:\Pyprogram\Python_Data_Analysis\data_csv\wired_data\wired_point_no_50294D201003011.csv"
# data_file = r"D:\Pyprogram\Python_Data_Analysis\data_csv\wireless_data\wireless_point_no_0a1ee4feff8d79e0-Z.csv"
id = re.search(r".*?_point_no_(.*?).csv", data_file).group(1)
print(id)
model_path = r'model\temperature\_'+id+'_lstm_model.h5'
ploty_html_path = r'html\_'+id+'_my_plot.html'

# 获取配置文件
config = get_config_from_json(json_file)


# 读取数据 D:\Pyprogram\Python_Data_Analysis\data_csv\temperature\Temperature_point_no_0a1ee4feff8d79e0-S.csv
# D:\Pyprogram\Python_Data_Analysis\data_csv\wired_data\wired_point_no_50294D201003011.csv
data = pd.read_csv(data_file, index_col=0)

# 转换数据格式
scaler = MinMaxScaler() #归一化
data = scaler.fit_transform(data)
# 将数据拆分为训练集和测试集
train_size = int(len(data) * 0.7)
train_data = data[:train_size, :]
test_data = data[train_size:, :]

# 创建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# 训练LSTM模型
model.fit(train_data[:, :, np.newaxis], train_data[:, 0], epochs=config["lstm_epochs"], batch_size=config["lstm_batch_size"])
model.save(model_path)
#########################################################


loaded_model = load_model(model_path)
# 对测试集进行预测
test_predict = loaded_model.predict(test_data[:, :, np.newaxis])
test_predict = np.squeeze(test_predict)

# 计算平均绝对误差
mae = np.mean(np.abs(test_predict - test_data[:, 0]))

# 根据平均绝对误差确定异常点
threshold = mae * config["lstm_threshold_mea_k"]
anomalies = np.where(np.abs(test_predict - test_data[:, 0]) > threshold)


# 反归一化数据
original_data = scaler.inverse_transform(data)
anomalies_data = scaler.inverse_transform(test_data[anomalies[0], :])

# 绘制原始数据和异常点

fig = plot_show_plotly(original_data,anomalies_data,anomalies,train_size)
fig.write_html(ploty_html_path)
# 输出异常值

anomalies_list = list(anomalies)
print(anomalies_list)