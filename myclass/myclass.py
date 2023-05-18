from pydantic import BaseModel
from typing import List
class TemperatureInput(BaseModel):
    """
    temperature json input
    look like datasets/json/more-uni.json
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

class VibrationInput(BaseModel):
    """
    VibrationInput json input
    look like datasets/json/more-multi.json
    """
    id: str
    dimension: int
    timestamps: List[int]
    valueNameList: List[str]
    values: List[List[float]]

class VibrationAnomalyOutput(BaseModel):
    id: str
    dimension: int
    anomalyLabelSum: List[int]

class VibrationRepairedOutput(BaseModel):
    id: str
    dimension: int
    anomalyLabelSum: List[int]
    repairedValues: List[List[float]]



# 将标记数据转换回json格式
def return_TemperatureAnomalyOutput_response(id,anomaly_label):
    response_data = {"id":id, "anomalyLabel": anomaly_label}
    response = TemperatureAnomalyOutput(**response_data)
    return  response.dict()
def return_TemperatureRepairedOutput_response(id,anomaly_label,repairedValues):
    response_data = {"id": id, "anomalyLabel": anomaly_label, "repairedValues": repairedValues}
    response = TemperatureRepairedOutput(**response_data)
    return  response.dict()
def return_VibrationAnomalyOutput_response(id,dimension,anomalyLabelSum):
    response_data = {"id": id, "dimension": dimension, "anomalyLabelSum": anomalyLabelSum}
    response = VibrationAnomalyOutput(**response_data)
    return  response.dict()
def return_VibrationRepairedOutput_response(id,dimension,anomalyLabelSum,repairedValues):
    response_data = {"id": id, "dimension": dimension, "anomalyLabelSum": anomalyLabelSum,"repairedValues": repairedValues}
    response = VibrationRepairedOutput(**response_data)
    return  response.dict()