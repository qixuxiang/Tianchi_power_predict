# coding: utf-8
'''
xgboost log model
'''

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from util import Path, used_features
from models.losses import evalmape, mape

np.random.seed(42)

def run1():
    dataset_store = pd.HDFStore(Path.dataset_path + '/dataset_0609.h5')
    train = dataset_store['train']
    train_X = train.drop(['total', 'persentage'], axis=1)
    train_y = train['persentage']
    test_X = dataset_store['test'][used_features]
    test_X_base = test_X['month_avg']

    dtrain = xgb.DMatrix(data=train_X, label=train_y, feature_names=train_X.columns)
    dtest = xgb.DMatrix(data=test_X, feature_names=test_X.columns)

    param = {'booster': 'gbtree', #gbtree',
             'max_depth': 6, 
             'min_child_weight': 2,
             'learning_rate': 0.05, # 0.03
             'subsample': 1,
             'colsample_bytree': 1,
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

def run():
    dataset_store = pd.HDFStore(Path.dataset_path + 'dataset_0610.h5')
    train = dataset_store['train']
    train_X = train.drop(['total'], axis=1)
    train_y = train['total']
    test_X = dataset_store['test']

    dtrain = xgb.DMatrix(data=train_X, label=train_y, feature_names=train_X.columns)
    dtest = xgb.DMatrix(data=test_X, feature_names=test_X.columns)

    param = {'booster': 'gbtree', #gbtree','dart'
             'max_depth': 6, 
             'min_child_weight': 2,
             'learning_rate': 0.05, # 0.03
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'objective': 'reg:linear',
             'silent': 1,
             'lambda': 1.0,
             'alpha': 0.8
             }

    evallist = [(dtrain, 'train')]
    num_round = 200
    bst = xgb.train(param, dtrain, num_round, evallist,  early_stopping_rounds=20) # feval=evalmape,
   
    fi_weight = bst.get_score(importance_type='weight')
    print("feature importance:")
    print(fi_weight)

    pred = bst.predict(dtest)

    #result = np.exp(pred).round().astype(int)
    #result = (pred+1)*test_X_base.values
    result = pred.round().astype(int)
    dataset_store.close()

    return result

def run3():
    dataset_store = pd.HDFStore(Path.dataset_path + 'dataset_0609.h5')
    train = dataset_store['train']
    train_X = train.drop(['total'], axis=1)
    train_y = train['total']
    test_X = dataset_store['test']

    #dtrain = xgb.DMatrix(data=train_X, label=train_y, feature_names=train_X.columns)
    #dtest = xgb.DMatrix(data=test_X, feature_names=test_X.columns)

    parameters = {'n_estimators': [140],
                  'max_depth': [4],
                  'subsample': [0.8],
                  'colsample_bytree': [0.9],
                  'learning_rate': [0.1],
                  'gamma': [2],
                  'min_child_weight': [2],
                  'reg_alpha': [0.4],
                  'reg_lambda': [0.4],
    }

    scoring = make_scorer(mape, greater_is_better=False)

    xgb_model = XGBRegressor()
   
    clf = GridSearchCV(estimator=xgb_model,
                       param_grid=parameters, scoring=scoring, cv=3, verbose=1)
    clf.fit(train_X, train_y)
    print(clf.best_score_)
    print(clf.best_params_)

    pred = clf.predict(test_X)
    #result = np.exp(pred).round().astype(int)
    #result = (pred+1)*test_X_base.values
    result = pred.round().astype(int)
    dataset_store.close()

    return result