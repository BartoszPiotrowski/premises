import os
import numpy as np
import xgboost as xgb
import tensorflow as tf
from .utils import printline, make_path
from .construct_network import Network


def train(labels, array, params={}, n_jobs=4, model_dir='',
          verbose=True, logfile=''):
    assert isinstance(labels, list)
    params['model'] = 'xgboost' if 'model' not in params else params['model']
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if params['model'] == 'xgboost':
        model_path = train_xgboost(labels, array, params, model_dir, n_jobs,
                                   verbose, logfile)
    elif params['model'] == 'network':
        model_path = train_network(labels, array, params, model_dir, n_jobs,
                                   verbose, logfile)
    else:
        raise ValueError("Model {} not available!".format(params['model']))
    return model_path


def train_xgboost(labels, array, params, model_dir, n_jobs, verbose, logfile):
    num_boost_round = params['num_boost_round'] \
        if 'num_boost_round' in params else 5000
    eta = params['eta'] if 'eta' in params else 0.1
    max_depth = params['max_depth'] if 'max_depth' in params else 10
    booster = params['booster'] if 'booster' in params else 'gbtree'
    assert booster in {'gbtree', 'gblinear', 'dart'}
    # 'gblinear' and 'dart' also available;
    # 'gblinear' makes xgboost like lasso
    pretrained_model = params['pretrained_model'] \
        if 'pretrained_model' in params else None
    dtrain = xgb.DMatrix(array, label=labels)
    params_booster = {'eta': eta,
                      'max_depth': max_depth,
                      'objective': 'binary:logistic',
                      'booster': booster,
                      'n_jobs': n_jobs}
    if verbose or logfile:
        printline("Training of xgboost model started...", logfile, verbose)
    model = xgb.train(params_booster, dtrain, num_boost_round=num_boost_round,
                      xgb_model=pretrained_model)
    model_path = make_path(model_dir, params)
    if verbose or logfile:
        printline("Training finished.", logfile, verbose)
        printline("Saving model to file {}".format(model_path),
                  logfile, verbose)
    model.save_model(model_path)
    return model_path


def train_network(labels, array, params, model_dir, n_jobs, verbose, logfile):
    from .construct_network import Network
    if verbose or logfile:
        printline("Constructing neural net graph...", logfile, verbose)
        network = Network(threads=n_jobs)
        network.construct(params)
    if verbose or logfile:
        printline("Training of neural net started...", logfile, verbose)
    for i in range(params['epochs']):
        printline('Epoch: {}'.format(i), logfile)
        indices = range(len(labels))
        permut_indices = np.random.permutation(indices)
        bs = params['batch_size']
        for j in range(len(indices) // bs):
            batch_indices = permut_indices[j * bs : (j + 1) * bs]
            array_batch = array[batch_indices].toarray()
            labels_batch = [labels[i] for i in batch_indices]
            network.train(array_batch, labels_batch)
        #network.evaluate("train", array_batch, labels_batch)
        batch_indices = np.random.choice(len(labels), params['batch_size'])
        array_batch = array[batch_indices].toarray()
        labels_batch = [labels[i] for i in batch_indices]
        accuracy = network.evaluate_accuracy(array_batch, labels_batch)
        printline(
            "Accuracy on random batch: {:.2f}".format(accuracy * 100), logfile)
        model_path = make_path(model_dir, params)
    if verbose or logfile:
        printline("Training finished.", logfile, verbose)
        printline("Saving model to file {}".format(model_path),
                  logfile, verbose)
    return network.save(model_path)
