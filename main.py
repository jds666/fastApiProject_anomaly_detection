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
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import load_model
from lstm.utils import *
from lstm.ploty_show import *
from lstm.useJson import *
from myclass.myclass import *
from lof.plotly_show import *
from ksigma.tocsv import *
from ksigma.useJson import *
from boxplot.useJson import *
from lof.useJson import *


app = FastAPI()

config = get_config_from_json(json_file)
# 均值计算


##########################################################################################
########################################## Finish ########################################
##########################################################################################


#####################################  csv-k-sigma  ######################################
@app.get('/anomaly_detection/temperature/k_sigma')
def anomaly_detection_temperature_k_sigma(path: str,k = config["k"]):
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/temperature/raw'
    anomalyDetectionTemperatureKSigma(path, k)
    return {"message:": "success"}

@app.get('/repair/temperature/k_sigma')
def repair_temperature_k_sigma(path: str,k = config["k"]):
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/temperature/raw'
    repairTemperatureKSigma(path, k)
    return {"message:": "success"}

@app.get('/anomaly_detection/vibration/k_sigma')
def anomaly_detection_vibration_k_sigma(path: str,k = config["k"], halfdaynum = 144):
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/vibration/raw'
    anomalyDetectionVibrationKSigma(path,k, halfdaynum)
    return {"message:": "success"}

@app.get('/repair/vibration/k_sigma')
def repair_vibration_k_sigma(path: str,k = config["k"], halfdaynum = 144):
    # 获取数据
    # path = 'D:/Pyprogram/fastApiProject_anomaly_detection/datasets/vibration/raw'
    repairVibrationKSigma(path,k, halfdaynum)
    return {"message:": "success"}


#####################################  json-k-sigma ######################################
@app.post("/Json/anomaly_detection/temperature/k-sigma")
def Json_anomaly_detection_temperature_k_sigma(Data: TemperatureInput, k = config["k"]):
    anomalyLabel = JsonAnomalyDetectionTemperatureKSigma(Data, k)
    return return_TemperatureAnomalyOutput_response(Data.id, anomalyLabel)


@app.post("/Json/anomaly_detection/vibration/k-sigma")
def Json_anomaly_detection_vibration_k_sigma(Data: VibrationInput,k = config["k"]):
    anomalyLabelSum = JsonAnomalyDetectionVibrationKSigma(Data, k)
    return return_VibrationAnomalyOutput_response(Data.id, Data.dimension, anomalyLabelSum)


@app.post("/Json/repair/temperature/k-sigma")
def Json_repair_temperature_k_sigma(Data: TemperatureInput,k= config["k"]):
    anomalyLabel, repairedData  = JsonRepairTemperatureKSigma(Data,k)
    return return_TemperatureRepairedOutput_response(Data.id, anomalyLabel, repairedData)


@app.post("/Json/repair/vibration/k-sigma")
def Json_repair_vibration_k_sigma(Data: VibrationInput,k = config["k"]):
    anomalyLabel, repairedData = JsonRepairVibrationKSigma(Data, k)
    return return_VibrationRepairedOutput_response(Data.id, Data.dimension, anomalyLabel, repairedData)


#####################################  json-boxplot ######################################
@app.post("/Json/anomaly_detection/temperature/boxplot")
def Json_anomaly_detection_temperature_boxplot(Data: TemperatureInput, k = config["k"]):
    anomalyLabel = JsonAnomalyDetectionTemperatureBoxplot(Data, k)
    return return_TemperatureAnomalyOutput_response(Data.id, anomalyLabel)


@app.post("/Json/anomaly_detection/vibration/boxplot")
def Json_anomaly_detection_vibration_boxplot(Data: VibrationInput,k = config["k"]):
    anomalyLabelSum = JsonAnomalyDetectionVibrationBoxplot(Data, k)
    return return_VibrationAnomalyOutput_response(Data.id, Data.dimension,anomalyLabelSum)


@app.post("/Json/repair/temperature/boxplot")
def Json_repair_temperature_boxplot(Data: TemperatureInput,k= config["k"]):
    anomalyLabel, repairedData = JsonRepairTemperatureBoxplot(Data, k)
    return return_TemperatureRepairedOutput_response(Data.id, anomalyLabel, repairedData)


@app.post("/Json/repair/vibration/boxplot")
def Json_repair_vibration_boxplot(Data: VibrationInput,k = config["k"]):
    anomalyLabelSum, repairedData = JsonRepairVibrationBoxplot(Data, k)
    return return_VibrationRepairedOutput_response(Data.id, Data.dimension, anomalyLabelSum, repairedData)


#####################################  json-lstm ######################################
@app.post("/Json/anomaly_detection/temperature/lstm")
def Json_anomaly_detection_temperature_lstm(Data: TemperatureInput):
    anomalyLabel = JsonAnomalyDetectionTemperatureLstm(Data)
    return return_TemperatureAnomalyOutput_response(Data.id, anomalyLabel)


@app.post("/Json/repair/temperature/lstm")
def Json_repair_temperature_lstm(Data: TemperatureInput):
    anomalyLabel, repairData = JsonRepairTemperatureLstm(Data)
    return return_TemperatureRepairedOutput_response(Data.id, anomalyLabel, repairData)


@app.post("/Json/anomaly_detection/vibration/lstm")
def Json_anomaly_detection_vibration_lstm(Data: VibrationInput):
    anomalyLabelSum = JsonAnomalyDetectionVibrationLstm(Data)
    return return_VibrationAnomalyOutput_response(Data.id, Data.dimension,anomalyLabelSum)


@app.post("/Json/repair/vibration/lstm")
def Json_repair_vibration_lstm(Data: VibrationInput):
    anomalyLabelSum, repairedData = JsonRepairVibrationLstm(Data)
    return return_VibrationRepairedOutput_response(Data.id, Data.dimension, anomalyLabelSum, repairedData)


#####################################  json-lof ######################################

@app.post("/Json/anomaly_detection/temperature/lof")
def Json_anomaly_detection_temperature_lof(Data: TemperatureInput):
    anomalyLabel = JsonAnomalyDetectionTemperatureLof(Data)
    return return_TemperatureAnomalyOutput_response(Data.id, anomalyLabel)


@app.post("/Json/anomaly_detection/vibration/lof")
def Json_anomaly_detection_vibration_lof(Data: VibrationInput):
    anomalyLabelSum = JsonAnomalyDetectionVibrationLof(Data)
    return return_TemperatureAnomalyOutput_response(Data.id, anomalyLabelSum)

@app.post("/Json/repair/temperature/lof")
def Json_repair_temperature_lof(Data: TemperatureInput):
    anomalyLabel, repairData = JsonRepairTemperatureLof(Data)
    return return_TemperatureRepairedOutput_response(Data.id, anomalyLabel, repairData)


# 主函数
if __name__ == "__main__":
    uvicorn.run(app='main:app', host='127.0.0.1', port=8181, reload=False)
