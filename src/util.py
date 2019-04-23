# coding: utf-8

import os
import pandas as pd

# 路径规定
current_path = os.path.abspath(__file__)
data_path = os.path.abspath(current_path + "/../../data/")

# 训练集：
train_day_time1 = pd.date_range(start='2015-04-01', end='2015-12-31', freq='D')
train_day_time2 = pd.date_range(start='2016-05-01', end='2016-09-30', freq='D')
validate_day_time1 = pd.date_range(start='2015-09-01', end='2015-10-31', freq='D')
validate_day_time2 = pd.date_range(start='2016-09-01', end='2016-09-30', freq='D')


class Path:
    data_file = data_path + "/data.csv"
    h5_data = data_path + "/data.h5"
    sub_path = os.path.abspath(current_path + "/../../submit/")
    hdf5_store = data_path + '/hdf5/'
    date_file = data_path + "/date.csv"
    dataset_path = data_path + '/dataset/'

    data_08 = data_path + "/Tianchi_power.csv"
    data_09 = data_path + "/Tianchi_power_9.csv"

class Date:
    # 训练集时间
    train_day_time = train_day_time1.append(train_day_time2)
    # 预测集时间
    predict_day_time = pd.date_range(start='20161001', end='20161031')
    # 验证时间
    validate_day_time = validate_day_time1.append(validate_day_time2)

# 模型映射
model_dict={
    'xgb': 'models.xgb_model_day',
    'gru': 'models.gru_model',
    'rf': 'models.rf_model_log',
    'gbdt': 'models.gbdt_model_log', 
}


used_features=['day_in_week', 'is_holiday', 'holiday_after', 'holiday_before', 'week_in_month', 'week_in_year', 'day_in_month', 'base14', 'month_avg', 'is_special_holiday', 'day_in_year', 'day_in_holiday', 'is_adjust']




