import xgboost as xgb
import numpy as np
from time import time
from joblib import Parallel, delayed

def train(labels, array, weights=None, model="xgboost", params={}, n_jobs=-1):
    assert isinstance(labels, list)
    assert isinstance(array, np.ndarray)
    if model == "xgboost":
        return train_xgboost(labels, array, weights, params, n_jobs)

def train_xgboost(labels, array, weights, params, n_jobs):
    num_boost_round = params["num_boost_round"] \
            if "num_boost_round" in params else 100
    eta = params["eta"] if "eta" in params else 0.05
    max_depth = params["max_depth"] if "max_depth" in params else 10
    xgb_pretrained_model = params["xgb_pretrained_model"] \
            if "xgb_pretrained_model" in params else None
    dtrain = xgb.DMatrix(array, label=labels, weight=weights)
    params_booster = {'eta': eta, 'max_depth': max_depth,
                      'objective': 'binary:logistic', 'n_jobs': n_jobs}
    return xgb.train(params_booster, dtrain, num_boost_round=num_boost_round,
                     xgb_model=xgb_pretrained_model)
