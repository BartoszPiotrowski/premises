import os
import numpy as np
import xgboost as xgb
import tensorflow as tf
from .utils import printline, make_path


def train(labels, array, labels_valid=None, array_valid=None, params={},
          pretrained_model_path=None, n_jobs=4, model_dir='models',
          verbose=True, logdir='', logfile=''):
    assert isinstance(labels, list)
    params['model'] = 'xgboost' if 'model' not in params else params['model']
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if logdir:
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        logdir = make_path(logdir, params)
    if params['model'] == 'xgboost':
        model_path = train_xgboost(labels, array, labels_valid, array_valid,
            params, pretrained_model_path, model_dir, n_jobs, verbose, logdir, logfile)
    elif params['model'] == 'network':
        model_path = train_network(labels, array, labels_valid, array_valid,
            params, pretrained_model_path, model_dir, n_jobs, verbose, logdir, logfile)
    else:
        raise ValueError("Model {} not available!".format(params['model']))
    return model_path


def train_xgboost(labels, array, labels_valid, array_valid, params,
                  pretrained_model_path, model_dir, n_jobs, verbose, logdir, logfile):
    num_boost_round = params['num_boost_round'] \
        if 'num_boost_round' in params else 5000
    eta = params['eta'] if 'eta' in params else 0.1
    max_depth = params['max_depth'] if 'max_depth' in params else 10
    booster = params['booster'] if 'booster' in params else 'gbtree'
    assert booster in {'gbtree', 'gblinear', 'dart'}
    # 'gblinear' and 'dart' also available;
    # 'gblinear' makes xgboost like lasso
    dtrain = xgb.DMatrix(array, label=labels)
    params_booster = {'eta': eta,
                      'max_depth': max_depth,
                      'objective': 'binary:logistic',
                      'booster': booster,
                      'n_jobs': n_jobs}
    if verbose or logfile:
        printline("Training of xgboost model started...", logfile, verbose)
    model = xgb.train(params_booster, dtrain, num_boost_round=num_boost_round,
                      xgb_model=pretrained_model_path)
    if verbose or logfile:
        printline("Training finished.", logfile, verbose)
    model_path = make_path(model_dir, params)
    if verbose or logfile:
        printline("Saving model to file {}".format(model_path),
                  logfile, verbose)
        model.save_model(model_path)
    return model_path


def train_network(labels, array, labels_valid, array_valid, params,
                  pretrained_model_path, model_dir, n_jobs, verbose, logdir, logfile):
    if not 'dual' in params:
        params['dual'] = False
    if params['dual']:
        from .construct_network_dual import Network
    else:
        from .construct_network import Network
    if pretrained_model_path:
        if verbose or logfile:
            printline("Restoring neural net graph...", logfile, verbose)
        network = Network(threads=n_jobs)
        network.load_and_train(pretrained_model_path, logdir)
    else:
        if verbose or logfile:
            printline("Constructing neural net graph...", logfile, verbose)
            network = Network(threads=n_jobs)
            network.construct(params, logdir)
    if verbose or logfile:
        printline("Training of neural net started...", logfile, verbose)
    for i in range(params['epochs']):
        printline("Epoch: {}".format(i), logfile)
        indices = range(len(labels))
        permut_indices = np.random.permutation(indices)
        bs = params['batch_size']
        for j in range(len(indices) // bs):
            batch_indices = permut_indices[j * bs : (j + 1) * bs]
            labels_batch = [labels[i] for i in batch_indices]
            if params['dual']:
                array_batch = [array[0][batch_indices].toarray(),
                               array[1][batch_indices].toarray()]
            else:
                array_batch = array[batch_indices].toarray()
            network.train(array_batch, labels_batch)
        # How is it going on training set?
        batch_indices = np.random.choice(len(labels), params['batch_size'])
        if params['dual']:
            array_batch = [array[0][batch_indices].toarray(),
                           array[1][batch_indices].toarray()]
        else:
            array_batch = array[batch_indices].toarray()
        labels_batch = [labels[i] for i in batch_indices]
        accuracy = network.evaluate_accuracy(array_batch, labels_batch)
        recall = network.evaluate_recall(array_batch, labels_batch)
        precision = network.evaluate_precision(array_batch, labels_batch)
        f1_score = network.evaluate_f1_score(array_batch, labels_batch)
        printline(
            "Accuracy/Precision/Recall/F1-score on a random training batch: "
            "  {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                accuracy, precision, recall, f1_score), logfile)
        if not labels_valid == None and not array_valid == None:
            # How is it going on validation set?
            batch_indices = np.random.choice(
                len(labels_valid), params['batch_size'])
            if params['dual']:
                array_batch = [array_valid[0][batch_indices].toarray(),
                               array_valid[1][batch_indices].toarray()]
            else:
                array_batch = array_valid[batch_indices].toarray()
            labels_batch = [labels_valid[i] for i in batch_indices]
            #network.evaluate_summaries(array_batch, labels_batch)
            accuracy = network.evaluate_accuracy(array_batch, labels_batch)
            recall = network.evaluate_recall(array_batch, labels_batch)
            precision = network.evaluate_precision(array_batch, labels_batch)
            f1_score = network.evaluate_f1_score(array_batch, labels_batch)
            printline(
            "Accuracy/Precision/Recall/F1-score on a random validation batch: "
                "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                    accuracy, precision, recall, f1_score), logfile)
        model_path = make_path(model_dir, params)
    if verbose or logfile:
        printline("Training finished.", logfile, verbose)
        printline("Saving model to file {}".format(model_path),
                  logfile, verbose)
    return network.save(model_path)
