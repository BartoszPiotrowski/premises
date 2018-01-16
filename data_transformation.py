from xgboost import DMatrix
from joblib import Parallel, delayed
from random import sample
import numpy as np
import scipy.sparse as sps
from time import time
from .utils import printline


def bin_vector(features_ordered, list_of_features):
#   return [int(i in list_of_features) for i in features_ordered]
# is it OK to have bools instead of ints?
    return [i in list_of_features for i in features_ordered]

def bin_trans_comb(thm_features, prm_features, features_ordered):
  #  vector = []
  #  for f in features_ordered:
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
                      for f in features_ordered]
  #  t1 = time(); print("1", t1 - t0)
    return vector

# TODO add other posibilities of transforming pair to vector
# pair means here (thm features, prm features)
def pairs_to_array(pairs, params):
    features_ordered = params['features_ordered']
    sparse = params['sparse'] if 'sparse' in params else False
    bin_vectors_trans = [bin_trans_comb(thm_f, prm_f, features_ordered)
                            for thm_f, prm_f in pairs]
    if sparse:
        # TODO to test
        return sps.coo_matrix(np.array(bin_vectors_trans))
    return np.array(bin_vectors_trans)

def proofs_to_train_one_theorem(thm, atp_useful, params_data_trans,
                                params_negative_mining):
    features = params_data_trans['features']
    features_ordered = params_data_trans['features_ordered']
    chronology = params_data_trans['chronology']
    ratio_neg_pos = params_data_trans['ratio_neg_pos'] \
        if 'ratio_neg_pos' in params_data_trans else 4
    sparse = params_data_trans['sparse'] if 'sparse' in params_data_trans \
        else False
    available_premises = chronology.available_premises(thm)
    # TODO here parameter about comb/concat/...
    not_positive_premises = set(available_premises) - atp_useful
    # TODO something more clever; differentiate importance of positives
    positive_premises = atp_useful
    if not len(not_positive_premises) > 1:
        return ([1] * len(positive_premises),
           pairs_to_array([(features[thm], features[prm])
                               for prm in positive_premises], params))
    negative_premises_misclassified = misclassified_negatives(
            params_negative_mining['rankings'][thm], atp_useful) \
            if params_negative_mining else set()
    negative_premises = \
           negative_premises_misclassified | set(sample(not_positive_premises,
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
    assert 'features_ordered' in params_data_trans
    assert 'chronology' in params_data_trans
    sparse = params_data_trans['sparse'] if 'sparse' in params_data_trans \
                                        else False
    if verbose or logfile:
        printline("Transforming proofs into training data...", logfile, verbose)
    with Parallel(n_jobs=n_jobs) as parallel:
        d_proofs_to_train_one_theorem = delayed(proofs_to_train_one_theorem)
        labels_and_arrays = parallel(
            d_proofs_to_train_one_theorem(thm, proofs.union_of_proofs(thm),
                                      params_data_trans, params_negative_mining)
                                         for thm in proofs)
    labels = [i for p in labels_and_arrays for i in p[0]]
    arrays = [p[1] for p in labels_and_arrays]
    if sparse:
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

