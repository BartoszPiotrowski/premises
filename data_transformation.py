from xgboost import DMatrix
from joblib import Parallel, delayed
from random import sample
import numpy as np
import scipy.sparse as sps
from time import time
from .utils import printline


def bin_vector(list_of_features, order_of_features):
#   return [int(i in list_of_features) for i in order_of_features]
# is it OK to have bools instead of ints?
    return [i in list_of_features for i in order_of_features]

def bin_trans_comb(thm_features, prm_features, order_of_features):
  #  vector = []
  #  for f in order_of_features:
  #      T = f in thm_features
  #      P = f in prm_features
  #      if T or P:
  #          if T and P:
  #              vector.append(1)
  #          else:
  #              vector.append(-1)
  #      else:
  #          vector.append(0)
  #  t0 = time()
    vector = [0 if not (f in thm_features or f in prm_features) else \
              (1 if f in thm_features and f in prm_features else -1)
                      for f in order_of_features]
  #  t1 = time(); print("1", t1 - t0)
    return vector

def bin_trans_concat(thm_features, prm_features, order_of_features):
    vector_thm = bin_vector(thm_features, order_of_features)
    vector_prm = bin_vector(prm_features, order_of_features)
    return vector_thm + vector_prm

# pair means here (thm features, prm features)
def pairs_to_array(pairs, params):
    order_of_features = params['features'].order_of_features
    sparse = params['sparse'] if 'sparse' in params else False
    merge_mode = params['merge_mode'] if 'merge_mode' in params else 'comb'
    if merge_mode == 'comb':
        bin_vectors_trans = [bin_trans_comb(thm_f, prm_f, order_of_features)
                                for thm_f, prm_f in pairs]
    if merge_mode == 'concat':
        bin_vectors_trans = [bin_trans_concat(thm_f, prm_f, order_of_features)
                                for thm_f, prm_f in pairs]
    if sparse:
        # TODO to test; DONE, but very slow... TODO make it faster
        return sps.coo_matrix(np.array(bin_vectors_trans))
    return np.array(bin_vectors_trans)

def proofs_to_train_one_theorem(thm, atp_useful, params_data_trans,
                                params_negative_mining):
    features = params_data_trans['features']
    chronology = params_data_trans['chronology']
    ratio_neg_pos = params_data_trans['ratio_neg_pos'] \
        if 'ratio_neg_pos' in params_data_trans else 4
    sparse = params_data_trans['sparse']
    available_premises = chronology.available_premises(thm)
    not_positive_premises = set(available_premises) - atp_useful
    # TODO something more clever; differentiate importance of positives
    positive_premises = atp_useful
    if not len(not_positive_premises) > 1:
        return ([1] * len(positive_premises),
           pairs_to_array([(features[thm], features[prm])
                           for prm in positive_premises], params_data_trans))
    negative_premises_misclassified = misclassified_negatives(
            params_negative_mining['rankings'][thm], atp_useful) \
            if params_negative_mining else set()
    negative_premises = \
        set(negative_premises_misclassified) | set(sample(not_positive_premises,
       min(len(not_positive_premises), ratio_neg_pos * len(positive_premises))))
    pairs_pos = [(features[thm], features[prm]) for prm in positive_premises]
    pairs_neg = [(features[thm], features[prm]) for prm in negative_premises]
    labels = [1] * len(pairs_pos) + [0] * len(pairs_neg)
    array = pairs_to_array(pairs_pos + pairs_neg, params_data_trans)
    assert len(labels) == array.shape[0]
    return labels, array

def proofs_to_train(proofs, params_data_trans, params_negative_mining={},
                    n_jobs=-1, verbose=True, logfile=''):
    assert 'features' in params_data_trans
    assert 'chronology' in params_data_trans
    if not 'sparse' in params_data_trans:
        params_data_trans['sparse'] = False
    if not 'merge_mode' in params_data_trans:
        params_data_trans['merge_mode'] = 'comb'
    if verbose or logfile:
        printline("Transforming proofs into training data...", logfile, verbose)
        printline("    merge_mode: {}".format(params_data_trans['merge_mode']),
                  logfile, verbose)
        printline("    sparse: {}".format(params_data_trans['sparse']),
                  logfile, verbose)
    with Parallel(n_jobs=n_jobs) as parallel:
        d_proofs_to_train_one_theorem = delayed(proofs_to_train_one_theorem)
        labels_and_arrays = parallel(
            d_proofs_to_train_one_theorem(thm, proofs.union_of_proofs(thm),
                                      params_data_trans, params_negative_mining)
                                         for thm in proofs)
    labels = [i for p in labels_and_arrays for i in p[0]]
    arrays = [p[1] for p in labels_and_arrays]
    if params_data_trans['sparse']:
        array = sps.vstack(arrays)
    else:
        array = np.concatenate(arrays)
    assert len(labels) == array.shape[0]
    if verbose or logfile:
        printline("Transformation finished.", logfile, verbose)
    return labels, array

# returns the most misclassified negatives
def misclassified_negatives(ranking, atp_useful, num_neg=2):
    n_pos = len(atp_useful)
    n_neg = int(n_pos * num_neg)
    mis_negs = [ranking[i] for i in range(min(n_neg, len(ranking)))
                if not ranking[i] in set(atp_useful)]
    return mis_negs

