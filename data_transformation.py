from xgboost import DMatrix
from joblib import Parallel, delayed
from random import sample
import numpy as np
import scipy.sparse as sps
#import gensim as gs
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
    sparse = params['sparse'] if 'sparse' in params else True
    merge_mode = params['merge_mode'] if 'merge_mode' in params else 'comb'
    if merge_mode == 'comb':
        bin_vectors_trans = [bin_trans_comb(thm_f, prm_f, order_of_features)
                                for thm_f, prm_f in pairs]
    if merge_mode == 'concat':
        bin_vectors_trans = [bin_trans_concat(thm_f, prm_f, order_of_features)
                                for thm_f, prm_f in pairs]
    if sparse:
        return sps.coo_matrix(np.array(bin_vectors_trans))
    return np.array(bin_vectors_trans)

def proofs_to_train_one_theorem(thm, atp_useful, params):
    features = params['features']
    chronology = params['chronology']
    sparse = params['sparse']
    available_premises = chronology.available_premises(thm)
    ratio_neg_pos = params['ratio_neg_pos'] \
        if 'ratio_neg_pos' in params else 5
    rankings_for_neg_mining = params['rankings_for_negative_mining'] \
        if 'rankings_for_negative_mining' in params else None
    level_of_neg_mining = params['level_of_negative_mining'] \
        if 'level_of_negative_mining' in params else 2
    not_pos_premises = set(available_premises) - set(atp_useful)
    # TODO something more clever; differentiate importance of positives
    pos_premises = atp_useful
    assert len(pos_premises) > 0
    if len(not_pos_premises) == 0:
        labels = [1] * len(pos_premises)
        array = pairs_to_array([(features[thm], features[prm])
                               for prm in pos_premises], params)
        assert len(labels) == array.shape[0]
        return labels, array
    if rankings_for_neg_mining:
        neg_premises_misclass = misclassified_negatives(
            rankings_for_neg_mining[thm], atp_useful, level_of_neg_mining)
        neg_premises_not_misclass = not_pos_premises - neg_premises_misclass
        num_neg_premises_not_misclass = \
            min(len(neg_premises_not_misclass), ratio_neg_pos * len(pos_premises))
        neg_premises_not_misclass_sample = \
         set(sample(neg_premises_not_misclass, num_neg_premises_not_misclass))
        neg_premises = neg_premises_misclass | neg_premises_not_misclass_sample
    else:
        num_neg = min(len(not_pos_premises), ratio_neg_pos * len(pos_premises))
        neg_premises = set(sample(not_pos_premises, num_neg))
    pairs_pos = [(features[thm], features[prm]) for prm in pos_premises]
    pairs_neg = [(features[thm], features[prm]) for prm in neg_premises]
    labels = [1] * len(pairs_pos) + [0] * len(pairs_neg)
    array = pairs_to_array(pairs_pos + pairs_neg, params)
    assert len(labels) == array.shape[0]
    return labels, array

def proofs_to_train(proofs, params, n_jobs=-1, verbose=True, logfile=''):
    assert len(proofs) > 0
    assert 'features' in params
    assert 'chronology' in params
    if not 'sparse' in params:
        params['sparse'] = True
    if not 'merge_mode' in params:
        params['merge_mode'] = 'comb'
    if 'rankings_for_negative_mining' in params:
        assert set(params['rankings_for_negative_mining']) >= set(proofs)
    if verbose or logfile:
        printline("Transforming proofs into training data...", logfile, verbose)
        printline("    Negative mining: {}".format(
                'rankings_for_negatve_mining' in params), logfile, verbose)
        printline(("    Mode of combining theorems and premises to "
                   "examples: merge_mode={}".format(params['merge_mode'])),
                  logfile, verbose)
    with Parallel(n_jobs=n_jobs) as parallel:
        d_proofs_to_train_one_theorem = delayed(proofs_to_train_one_theorem)
        labels_and_arrays = parallel(
            d_proofs_to_train_one_theorem(thm, proofs.union_of_proofs(thm),
                                      params) for thm in proofs)
    labels = [i for p in labels_and_arrays for i in p[0]]
    arrays = [p[1] for p in labels_and_arrays]
    if params['sparse']:
        array = sps.vstack(arrays)
    else:
        array = np.concatenate(arrays)
    assert len(labels) == array.shape[0]
    if verbose or logfile:
        printline("Transformation finished.", logfile, verbose)
    return labels, array

# returns the most misclassified negatives
def misclassified_negatives(ranking, atp_useful, level_of_neg_mining=2):
    if isinstance(level_of_neg_mining, int):
        n_pos = len(atp_useful)
        n_neg = int(n_pos * level_of_neg_mining)
        mis_negs = [ranking[i] for i in range(min(n_neg, len(ranking)))
                    if not ranking[i] in set(atp_useful)]
    elif level_of_neg_mining == 'all':
        max_pos = max([i if prm in atp_useful else 0
                    for i, prm in enumerate(ranking)])
        mis_negs = [ranking[i] for i in range(min(max_pos, len(ranking)))
                    if not ranking[i] in set(atp_useful)]
    elif level_of_neg_mining == 'random':
        max_pos = max([i if prm in atp_useful else 0
                    for i, prm in enumerate(ranking)])
        mis_negs_all = [ranking[i] for i in range(min(max_pos, len(ranking)))
                    if not ranking[i] in set(atp_useful)]
        mis_negs = sample(mis_negs_all, len(mis_negs_all) // 2)
    else:
        print("Error: no such level of negative mining.")
    return set(mis_negs)

