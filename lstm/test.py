import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from utils import *
from ploty_show import *


config = get_config_from_json(json_file)

#测试数据
data = pd.read_csv(test_dataset_path, index_col=0)
# 转换数据格式
scaler = MinMaxScaler()  # 归一化
data = scaler.fit_transform(data)
test_data = data[0:, :]

# 加载模型
loaded_model = load_model(test_model_path)
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

fig = plot_show_plotly(original_data,anomalies_data,anomalies,0)
ploty_html_path= create_ploty_html_path (test_dataset_path)
fig.write_html(ploty_html_path)
# 输出异常值

anomalies_list = list(anomalies)
print(anomalies_list)