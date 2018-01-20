from xgboost import DMatrix
from joblib import Parallel, delayed
from random import sample
import numpy as np
import scipy.sparse as sps
from sklearn.feature_extraction import FeatureHasher
#import gensim as gs
from time import time
from .utils import printline, partition

# pair means here (thm features, prm features)
def pairs_to_array(pairs, params):
    num_of_features = params['features'].num_of_features
    merge_mode = params['merge_mode']
    if merge_mode == 'comb':
        list_of_pairs = [list(thm_f) + list(prm_f) for thm_f, prm_f in pairs]
    else:
        num_of_features = 2 * num_of_features
        list_of_pairs = []
        for thm_f, prm_f in pairs:
            thm_f_appended = ['T' + f for f in thm_f]
            prm_f_appended = ['T' + f for f in prm_f]
            list_of_pairs.append(thm_f_appended + prm_f_appended)
    hasher = FeatureHasher(n_features=num_of_features, input_type='string')
    csc_array = hasher.transform(list_of_pairs)
    return csc_array

def proofs_to_train_n_thms(thms, proofs, params):
    labels, pairs = [], []
    for thm in thms:
        atp_useful = proofs.union_of_proofs(thm)
        labels_thm, pairs_thm = thm_to_labels_and_pairs(thm, atp_useful, params)
        labels.extend(labels_thm)
        pairs.extend(pairs_thm)
    array = pairs_to_array(pairs, params)
    return labels, array

def thm_to_labels_and_pairs(thm, atp_useful, params):
    features = params['features']
    chronology = params['chronology']
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
    if len(not_pos_premises) == 0:
        labels = [1] * len(pos_premises)
        pairs = [(features[thm], features[prm]) for prm in pos_premises]
        return labels, pairs
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
    pairs = pairs_pos + pairs_neg
    return labels, pairs

def proofs_to_train(proofs, params, n_jobs=-1, verbose=True, logfile=''):
    assert len(proofs) > 0
    assert 'features' in params
    assert 'chronology' in params
    if not 'merge_mode' in params:
        params['merge_mode'] = 'concat'
    if 'rankings_for_negative_mining' in params:
        assert set(params['rankings_for_negative_mining']) >= set(proofs)
    if verbose or logfile:
        printline("Transforming proofs into training data...", logfile, verbose)
        printline("    Negative mining: {}".format(
                'rankings_for_negatve_mining' in params), logfile, verbose)
        printline(("    Mode of combining thms and premises to "
                   "examples: merge_mode={}".format(params['merge_mode'])),
                  logfile, verbose)
    all_proved_thms = list(proofs)
    thms_splited = partition(all_proved_thms, n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        d_proofs_to_train_n_thms = delayed(proofs_to_train_n_thms)
        labels_and_arrays = parallel(
            d_proofs_to_train_n_thms(thms, proofs, params)
                        for thms in thms_splited)
    labels = [i for p in labels_and_arrays for i in p[0]]
    arrays = [p[1] for p in labels_and_arrays]
    array = sps.vstack(arrays)
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

