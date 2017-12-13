from xgboost import DMatrix
from joblib import Parallel, delayed
from random import sample
import numpy as np


def bin_vector(features_ordered, list_of_features):
#   return [int(i in list_of_features) for i in features_ordered]
    return [i in list_of_features for i in features_ordered]

def bin_trans_comb(thm_features, prm_features, features_ordered):
    vector = []
    for f in features_ordered:
        if f in thm_features or f in prm_features:
            if f in thm_features and f in prm_features:
                vector.append(1)
            else:
                vector.append(-1)
        else:
            vector.append(0)
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

def proofs_to_train_one_theorem(thm, proofs, params):
    features = params['features']
    features_ordered = params['features_ordered']
    chronology = params['chronology']
    ratio_neg_pos = params['ratio_neg_pos'] if 'ratio_neg_pos' in params else 4
    sparse = params['sparse'] if 'sparse' in params else False
    available_premises = chronology.available_premises(thm)
    # TODO here parameter about comb/concat/...
    not_positive_premises = set(available_premises) - set().union(*proofs)
    # TODO something more clever; differentiate importance of positives
    positive_premises = set().union(*proofs)
    if not len(not_positive_premises) > 1:
        return ([1] * len(positive_premises),
           pairs_to_array([(features[thm], features[prm])
                               for prm in positive_premises], params))
    negative_premises = set(sample(not_positive_premises,
                                   ratio_neg_pos * len(positive_premises)))
    pairs_pos = [(features[thm], features[prm]) for prm in positive_premises]
    pairs_neg = [(features[thm], features[prm]) for prm in negative_premises]
    labels = [1] * len(pairs_pos) + [0] * len(pairs_neg)
    array = pairs_to_array(pairs_pos + pairs_neg, params)
    assert len(labels) == array.shape[0]
    return labels, array

def proofs_to_train(proofs, params, n_jobs=-1):
    with Parallel(n_jobs=n_jobs) as parallel:
        d_proofs_to_train_one_theorem = delayed(proofs_to_train_one_theorem)
        labels_and_arrays = parallel(
            d_proofs_to_train_one_theorem(thm, proofs[thm], params)
                                     for thm in proofs)
    labels = [i for p in labels_and_arrays for i in p[0]]
    arrays = [p[1] for p in labels_and_arrays]
    array = np.concatenate(arrays)
    assert len(labels) == array.shape[0]
    return labels, array
