from xgboost import DMatrix


def bin_vector(list_of_all_features, list_of_features):
#   return [int(i in list_of_features) for i in list_of_all_features]
    return [i in list_of_features for i in list_of_all_features]

def bin_trans_comb(list_of_features, thm_features, prm_features):
    vector = []
    for f in list_of_features:
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
def pairs_to_array(pairs, list_of_features, sparse=False):
    bin_vectors_trans = [bin_trans_comb(list_of_features, thm_f, prm_f)
                            for thm_f, prm_f in pairs]
    if sparse:
        # TODO to test
        return sps.coo_matrix(np.array(bin_vectors_trans))
    return np.array(bin_vectors_trans)

