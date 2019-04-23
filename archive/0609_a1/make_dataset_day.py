# -*- coding: utf-8 -*-

import pandas as pd
from util import Path, Date

def make_dataset_0608():
    store = pd.HDFStore(Path.h5_data)
    dataset_store = pd.HDFStore(Path.dataset_path + "/dataset_0608.h5")
    power_raw = store['power_raw']['total']
    feature = store['date']

    #feature = pd.merge(date, power_raw, how='left', left_index=True, right_index=True)
    feature['total'] = store['power_raw']['total']
    feature['month_avg'] = store['power_month_avg']['total']
    feature['base14'] = store['power_month_last14']['total']
    feature['base7'] = store['power_month_last7']['total']
    feature['delta'] = feature['total'] - feature['month_avg']
    #feature['delta14'] = feature['total'] - feature['base14']
    #feature['delta7'] = feature['total'] - feature['base7']

    #feature['persentage7'] = feature['delta7'] / feature['base7']
    #feature['persentage14'] = feature['delta14'] / feature['base14']
    feature['persentage'] = feature['delta'] / feature['month_avg'] 

    feature = feature.drop(['delta'], axis=1)

    train = feature.loc[Date.train_day_time]
    test = feature.loc[Date.predict_day_time]
    test = test.drop(['total', 'persentage'], axis=1)

    # Save
    print("Save Data...")
    dataset_store['train'] = train
    dataset_store['test'] = test

    store.close()
    dataset_store.close()

def make_dataset_0609():
    store = pd.HDFStore(Path.h5_data)
    dataset_store = pd.HDFStore(Path.dataset_path + "/dataset_0609.h5")

    feature = store['date']
    feature['total'] = store['power_raw']['total']
    feature['month_avg'] = store['power_month_avg']['total']
    feature['base14'] = store['power_month_last14']['total']
    feature['base7'] = store['power_month_last7']['total']

    for day in feature.index[31:]:
        last_month_end = day - pd.offsets.MonthEnd()
        last_month_begin = last_month_end - pd.offsets.MonthBegin()

        time_range = pd.date_range(start=last_month_begin, end=last_month_end, freq='D')
        time_feature = feature.loc[time_range]
        today_is_holiday = feature.loc[day, 'is_holiday']

        days = time_range[time_range.weekday == day.weekday()]
        weekday_feature = time_feature.loc[days]
        subset = weekday_feature[weekday_feature['is_holiday'] == today_is_holiday]

        work_holi_day = time_feature[time_feature['is_holiday'] == today_is_holiday]['total']

        if len(subset) < 2:
            feature.loc[day, 'last_month_weekday_mean'] = feature['total'].loc[time_range].mean()
            feature.loc[day, 'last_month_weekday_median'] = feature['total'].loc[time_range].median()
        else:
            feature.loc[day, 'last_month_weekday_mean'] = subset['total'].loc[time_range].mean()
            feature.loc[day, 'last_month_weekday_median'] = subset['total'].loc[time_range].median()
        feature.loc[day, 'work_holi_day_mean'] = work_holi_day.mean()
        feature.loc[day, 'work_holi_day_median'] = work_holi_day.median()
        feature.loc[day, 'work_holi_day_var'] = work_holi_day.var()
        feature.loc[day, 'last_month_var'] = time_feature['total'].var()

    train = feature.loc[Date.train_day_time]
    test = feature.loc[Date.predict_day_time]
    test = test.drop(['total'], axis=1)

    # Save
    print("Save Data...")
    dataset_store['train'] = train
    dataset_store['test'] = test

    store.close()
    dataset_store.close()

def make_dataset_0610():
    store = pd.HDFStore(Path.h5_data)
    dataset_store = pd.HDFStore(Path.dataset_path + "/dataset_0610.h5")

    feature = store['date']
    #feature = feature.drop(['day_in_year'], axis=1) # , 'week_in_year'
    feature['total'] = store['power_raw']['total']
    feature['month_avg'] = store['power_month_avg']['total']
    feature['base14'] = store['power_month_last14']['total']
    feature['base7'] = store['power_month_last7']['total']

    for day in feature.index[31:]:
        last_month_end = day - pd.offsets.MonthEnd()
        last_month_begin = last_month_end - pd.offsets.MonthBegin()

        time_range = pd.date_range(start=last_month_begin, end=last_month_end, freq='D')
        time_feature = feature.loc[time_range]
        today_is_holiday = feature.loc[day, 'is_holiday']

        days = time_range[time_range.weekday == day.weekday()]
        weekday_feature = time_feature.loc[days]
        subset = weekday_feature[weekday_feature['is_holiday'] == today_is_holiday]

        work_holi_day = time_feature[time_feature['is_holiday'] == today_is_holiday]['total']

        if len(subset) < 2:
            feature.loc[day, 'last_month_weekday_mean'] = feature['total'].loc[time_range].mean()
            #feature.loc[day, 'last_month_weekday_median'] = feature['total'].loc[time_range].median()
        else:
            feature.loc[day, 'last_month_weekday_mean'] = subset['total'].loc[time_range].mean()
            #feature.loc[day, 'last_month_weekday_median'] = subset['total'].loc[time_range].median()
        feature.loc[day, 'work_holi_day_mean'] = work_holi_day.mean()
        #feature.loc[day, 'work_holi_day_median'] = work_holi_day.median()
        feature.loc[day, 'work_holi_day_var'] = work_holi_day.var()
        feature.loc[day, 'last_month_var'] = time_feature['total'].var()
        #feature.loc[day, 'last_month_skew'] = time_feature['total'].skew()
        #feature.loc[day, 'last_month_kurt'] = time_feature['total'].kurt()
        #feature.loc[day, 'last_month_mad'] = time_feature['total'].mad()
        #for i in range(1, 8):
        #    feature.loc[day, 'last_month_last_%d' % i] = time_feature['total'].iloc[-i]

    train = feature.loc[Date.train_day_time]
    test = feature.loc[Date.predict_day_time]
    test = test.drop(['total'], axis=1)

    # Save
    print("Save Data...")
    dataset_store['train'] = train
    dataset_store['test'] = test

    store.close()
    dataset_store.close()