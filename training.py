import xgboost as xgb
import numpy as np
from time import time
from joblib import Parallel, delayed

def train(labels, array, weights=None, model="xgboost", params={}, n_jobs=-1):
    if model == "xgboost":
        num_boost_round = params["num_boost_round"] \
                if "num_boost_round" in params else 100
        eta = params["eta"] if "eta" in params else 0.05
        max_depth = params["max_depth"] if "max_depth" in params else 10
        xgb_pretrained_model = params["xgb_pretrained_model"] \
                if "xgb_pretrained_model" in params else None
        return train_xgboost(labels, array, weights, num_boost_round, eta,
                             max_depth, xgb_pretrained_model, n_jobs)

def train_xgboost(labels, array, weights, num_boost_round, eta,
                  max_depth, xgb_pretrained_model, n_jobs):
    dtrain = xgb.DMatrix(array, label=labels, weight=weights)
    params = {'eta': eta, 'max_depth': max_depth,
              'objective': 'binary:logistic',
              'n_jobs': n_jobs}
    return xgb.train(params, dtrain, num_boost_round=num_boost_round,
                     xgb_model=xgb_pretrained_model)
