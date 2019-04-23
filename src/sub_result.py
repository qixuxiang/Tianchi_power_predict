# coding: utf-8
'''
提交代码
'''

import os
import sys
import datetime
import pandas as pd
from util import Path, Date, model_dict

def run(model="xgb"):
    sub = pd.DataFrame(columns=['predict_power_consumption'], index=Date.predict_day_time)
    model = model_dict.get(model, "models.xgb_model")
    model = __import__(model, fromlist=["models"])
    pred = model.run()
    print("均值：", pred.mean())
    sub['predict_power_consumption'] = pred.reshape(-1)
    print(sub)
    return sub

if __name__ == '__main__':
    if len(sys.argv) < 2:
        method = "xgb"
    else :
        method = sys.argv[1]
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(Path.sub_path + '/' + today):
        os.mkdir(Path.sub_path + '/' + today)
    sub_file = Path.sub_path + '/' + today + '/Tianchi_power_predict_table.csv'
    sub = run(method)
    sub.index = sub.index.strftime("%Y%m%d")
    sub.to_csv(sub_file, index_label='predict_date')