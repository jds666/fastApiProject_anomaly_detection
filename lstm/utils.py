import json
import re

json_file = r"D:\Pyprogram\fastApiProject_anomaly_detection\python_anomaly_detection.json"
test_model_path = 'model\\temperature\\01c476feff958edc-S_lstm_model.h5'
test_dataset_path = r'D:\Pyprogram\Python_Data_Analysis\data_csv\temperature\Temperature_point_no_0a1ee4feff8d79e0-S.csv'
train_datasets_path = "D:\Pyprogram\Python_Data_Analysis\data_csv"
def get_config_from_json(json_file):
  """
  Get the config from a json file
  :param json_file:
  :return: config(dictionary)
  """
  # parse the configurations from the config json file provided
  with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)

  return config_dict

def create_model_paths(file_path):
    id = re.search(r".*?_point_no_(.*?).csv", file_path).group(1)
    type = re.search(r'\\(\w+)\\[^\\]+$', file_path).group(1)
    model_path = f'model\\{type}\\{id}_lstm_model.h5'
    return model_path

def create_ploty_html_path(file_path):
    id = re.search(r".*?_point_no_(.*?).csv", file_path).group(1)
    type = re.search(r'\\(\w+)\\[^\\]+$', file_path).group(1)
    ploty_html_path = rf'html\\{type}\\{id}_my_plot.html'
    return ploty_html_path