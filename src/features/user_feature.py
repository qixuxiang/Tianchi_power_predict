# -*- coding: utf-8 -*-

import pandas as pd
from util import Path, Date

def make_feature():
    store = pd.HDFStore(Path.h5_data)
    power = store['power_raw']
    user_id = [str(i) for i in range(1, 1455)]
    power_train = power.loc[Date.train_day_time]
    power_train_10 = power.loc[Date.validate_day_time]

    corr = pd.DataFrame(columns=['corr'])
    for i in user_id:
        corr.loc[i] = power_train[i].corr(power_train['total'])
    corr_train = corr.sort_values(by='corr', ascending=False)

    corr_10 = pd.DataFrame(columns=['corr'])
    for i in user_id:
        corr_10.loc[i] = power_train_10[i].corr(power_train_10['total'])
    corr_10 = corr_10.sort_values(by='corr', ascending=False)

    power_ratio = power.copy()
    for i in user_id:
        power_ratio[i] = power_ratio[i] / power_ratio['total']
    ratio = power_ratio.drop('total', axis=1).mean()