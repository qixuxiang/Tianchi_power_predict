# coding: utf-8

import numpy as np

def mapeobj(preds,dtrain):
    gaps = dtrain.get_label()
    gaps = np.exp(gaps)
    preds = np.exp(preds)
    grad = np.sign(preds-gaps)/gaps
    hess = 1/gaps
    grad[(gaps==0)] = 0
    hess[(gaps==0)] = 0
    return grad,hess  

def evalmape(preds, dtrain):
    gaps = dtrain.get_label()
    err = abs(gaps-preds)/gaps
    err[(gaps==0)] = 0
    err = np.mean(err)
    return 'error',err 

def mape_log(y_true, y_pred):
    y_pred = np.exp(y_pred)
    y_true = np.exp(y_true)
    return np.mean((np.abs(y_true - y_pred) / y_true))

def mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true))

    
