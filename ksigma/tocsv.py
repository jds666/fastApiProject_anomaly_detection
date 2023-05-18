import numpy as np
import pandas as pd
import datetime
import os
import csv



def anomalyDetectionTemperatureKSigma(path, k):
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

def repairTemperatureKSigma(path, k):
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
                    # print(data.duplicated().sum())

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


                    # print(data.head())

                    # 修复结果保存到文件中
                    output_path = subdir_path.replace("raw\\", "output/")
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)

                    # 文件名
                    repaired_filename = os.path.join(output_path, 'repaired.csv')

                    data.to_csv(repaired_filename, index=False)

def anomalyDetectionVibrationKSigma(path,k, halfdaynum):
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
                    elif data.shape[1] == 13:
                        data.columns = ['timestamp','AccelerationPeakX', 'AccelerationRmsX', 'SpeedPeakX', 'SpeedRmsX',
                                        'AccelerationPeakY', 'AccelerationRmsY', 'SpeedPeakY', 'SpeedRmsY',
                                        'AccelerationPeakZ', 'AccelerationRmsZ', 'SpeedPeakZ', 'SpeedRmsZ']
                    else:
                        print("Error: Data file has unsupported number of columns. Exiting...")
                        return
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

def repairVibrationKSigma(path,k, halfdaynum):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            # sum_data = 0
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)

                # 读取文件到pandas
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',', nrows=halfdaynum)
                    # name for data,first index is timestamp,others are value
                    if data.shape[1] == 5:
                        data.columns = ['timestamp', 'AccelerationPeakX', 'AccelerationRmsX', 'SpeedPeakX', 'SpeedRmsX']
                    elif data.shape[1] == 13:
                        data.columns = ['timestamp', 'AccelerationPeakX', 'AccelerationRmsX', 'SpeedPeakX', 'SpeedRmsX',
                                        'AccelerationPeakY', 'AccelerationRmsY', 'SpeedPeakY', 'SpeedRmsY',
                                        'AccelerationPeakZ', 'AccelerationRmsZ', 'SpeedPeakZ', 'SpeedRmsZ']
                    else:
                        print("Error: Data file has unsupported number of columns. Exiting...")
                        return

                    # 时间戳转换成日期
                    data['timestamp'] = data['timestamp'].apply(
                        lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                    data['timestamp'] = pd.to_datetime(data['timestamp'])

                    anomaly_label = []
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
                    data.values[:, :] = np.array(repairedValues_T)
                    # 存入csv
                    output_path = subdir_path.replace("raw\\", "output/")
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)

                    repaired_filename = os.path.join(output_path, 'reparied')
                    data.to_csv(repaired_filename, index=False)
