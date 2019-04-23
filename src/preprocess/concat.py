# coding: utf-8

import pandas as pd
from util import Path, Date


def cancat_data():
    data_08 = pd.read_csv(Path.data_08, parse_dates=['record_date'])
    data_09 = pd.read_csv(Path.data_09, parse_dates=['record_date'])
    data_08_pivot = data_08.pivot(index='record_date', columns='user_id', values='power_consumption')
    data_08_pivot = data_08_pivot.fillna(data_08_pivot.mean().round())
    data_09_pivot = data_09.pivot(index='record_date', columns='user_id', values='power_consumption') 

    power = pd.concat([data_08_pivot, data_09_pivot])
    power['total'] = power.sum(axis=1)
    power = power.astype(int)
    power.to_csv(Path.data_file)
    store = pd.HDFStore(Path.h5_data)
    power = pd.read_csv(Path.data_file, index_col=0, parse_dates=[0])
    store['power_raw'] = power

    date = pd.read_csv(Path.date_file, parse_dates=[0], index_col=0)
    store['date'] = date

    store.close()

def base_line():
    store = pd.HDFStore(Path.h5_data)
    power_raw = store['power_raw']
    for date in Date.predict_day_time:
        power_raw.loc[date] = 0

    power_month_avg = power_raw.copy()
    power_month_last7 = power_raw.copy()
    power_month_last14 = power_raw.copy()
    # 上一个月的平均值: 7天， 14天， 月
    month_avg = power_raw.groupby(pd.TimeGrouper('M')).mean().shift(1)
    month_avg.index = month_avg.index.to_period()

    for index, month in enumerate(month_avg.index.to_native_types()):
        last_week_in_month = pd.to_datetime(month) - pd.offsets.MonthEnd()
        if last_week_in_month < pd.to_datetime('2015-01-01'):
            continue
       
        for day in pd.date_range(start=pd.to_datetime(month), end=pd.to_datetime(month) + pd.offsets.MonthEnd()):
            # 上个月平均值
            power_month_avg.loc[day] = month_avg.loc[month]
            # 上个月最后7天均值
            power_month_last7.loc[day] = power_raw.loc[pd.date_range(end=last_week_in_month, periods=7)].mean()
            # 上个月最后14天均值
            power_month_last14.loc[day] = power_raw.loc[pd.date_range(end=last_week_in_month, periods=14)].mean()
    
    store['power_month_avg'] = power_month_avg.round().astype(int)
    store['power_month_last7'] = power_month_last7.round().astype(int)
    store['power_month_last14'] = power_month_last14.round().astype(int)
    store.close()
        