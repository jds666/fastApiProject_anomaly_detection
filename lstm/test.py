import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from utils import *
from ploty_show import *


config = get_config_from_json(json_file)

#测试数据
data = pd.read_csv(test_dataset_path, index_col=0)
# print(data.head(5))
# 转换数据格式
scaler = MinMaxScaler()  # 归一化
Data = scaler.fit_transform(data)
# print(Data)
test_data = Data[0:, :]
# print(test_data)

# print(test_data.shape)
# 加载模型
# print(create_model_paths(test_dataset_path))
loaded_model = load_model(create_model_paths(test_dataset_path))

# print(loaded_model.summary())
# 对测试集进行预测
test_predict = loaded_model.predict(test_data[:, :, np.newaxis])
# print(test_predict.shape)

test_predict = np.squeeze(test_predict)

# print(test_predict.shape)

# 计算平均绝对误差
mae = np.mean(np.abs(test_predict - test_data[:, 0]))


# 根据平均绝对误差确定异常点
threshold = mae * config["lstm_threshold_mea_k"]


anomalies = np.where(np.abs(test_predict - test_data[:, 0]) > threshold)


# 反归一化数据
original_data = scaler.inverse_transform(Data)
# print(original_data)
# print(data.index.tolist())

# 绘制原始数据和异常点

fig = plot_show_plotly(original_data,anomalies,data.index.tolist(),0,"test")
ploty_html_path= create_ploty_html_path (test_dataset_path)
fig.write_html(ploty_html_path)

# 输出异常值
anomalies_list = list(anomalies)
print(anomalies_list)