import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sps
from time import time
from datetime import datetime
from joblib import Parallel, delayed
from .construct_net import Network
from .utils import printline

def train_net(labels, array, params={}, n_jobs=4, save_model_to_file=None,
              verbose=True, logfile=''):
    assert isinstance(labels, list)
    if verbose or logfile:
        printline("Constructing neural net graph...", logfile, verbose)
        network = Network(threads=n_jobs)
        network.construct(params)
    if verbose or logfile:
        printline("Training of neural net started...", logfile, verbose)
    for i in range(params['epochs']):
        printline('Iteration: {}'.format(i), logfile)
        batch_indices = np.random.choice(
            len(labels), params['batch_size'])
        array_batch = array[batch_indices].toarray()
        labels_batch = [labels[i] for i in batch_indices]
        network.train(array_batch, labels_batch)
        #network.evaluate("train", array_batch, labels_batch)
        acc = network.evaluate_accuracy('train', array_batch, labels_batch)
        printline("Accuracy on training batch: {:.2f}".format(acc * 100), logfile)
        model_path = os.path.join(
            params['model_dir'], datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    if verbose or logfile:
        printline("Training finished.", logfile, verbose)
        printline("Saving model to the file {}".format(model_path),
                  logfile, verbose)
    network.save(model_path)
    return model_path
