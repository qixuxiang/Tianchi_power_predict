# coding: utf-8
'''
xgboost log model
'''

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from util import Path
from models.losses import evalmape

np.random.seed(42)

def run():
    dataset_store = pd.HDFStore(Path.dataset_path + '/dataset_0608.h5')
    train = dataset_store['train']
    train_X = train.drop(['total', 'persentage'], axis=1)
    train_y = train['persentage']
    test_X = dataset_store['test']
    test_X_base = test_X['month_avg']

    dtrain = xgb.DMatrix(data=train_X, label=train_y, feature_names=train_X.columns)
    dtest = xgb.DMatrix(data=test_X, feature_names=test_X.columns)

    param = {'booster': 'dart', #gbtree',
             'max_depth': 6, 
             'min_child_weight': 2,
             'learning_rate': 0.05, # 0.03
             'subsample': 0.8,
             'colsample_bytree': 0.9,
             'objective': 'reg:linear',
             'silent': 1,
             'lambda': 0.4,
             'alpha': 0.4
             }

    evallist = [(dtrain, 'train')]
    num_round = 250
    bst = xgb.train(param, dtrain, num_round, evallist,  early_stopping_rounds=20) # feval=evalmape,
   
    fi_weight = bst.get_score(importance_type='weight')
    print("feature importance:")
    print(fi_weight)

    pred = bst.predict(dtest)

    #result = np.exp(pred).round().astype(int)
    result = (pred+1)*test_X_base.values
    result = result.round().astype(int)
    dataset_store.close()

    return result