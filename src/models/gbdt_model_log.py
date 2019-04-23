# -*- coding: utf-8 -*-

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from util import hdf5_store
from models.losses import mape_log
import pandas as pd
import numpy as np

np.random.seed(2017)

def run():
    dataset_store = pd.HDFStore(hdf5_store + '/dataset_log.h5')
    X_train = dataset_store['X_train']
    y_train = dataset_store['y_train']
    X_test = dataset_store['X_test']

    parameters = {
        'estimator__n_estimators': [100, 150],
        'estimator__loss': ['huber'],
        'estimator__learning_rate': [0.01, 0.02],
        'estimator__max_depth': [5, 6],
        'estimator__min_samples_split': [8, 10],
        'estimator__min_weight_fraction_leaf': [0.1, 0.2],
        'estimator__min_samples_leaf': [8, 10],
        'estimator__max_leaf_nodes':[8, 6],
        'estimator__max_features': [120, 150],
        'estimator__warm_start': [True], 
    }

    scoring = make_scorer(mape_log, greater_is_better=False)
    gbdt_model = MultiOutputRegressor(GradientBoostingRegressor())
    clf = RandomizedSearchCV(estimator=gbdt_model, param_distributions=parameters, n_iter=30)

    clf.fit(X_train.values, y_train.values)
    print(clf.best_score_)
    print(clf.best_params_)

    pred = clf.predict(X_test.values)

    dataset_store.close()
    result = np.exp(pred).round().astype(int)

    return result 