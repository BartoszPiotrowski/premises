from joblib import Parallel, delayed
from .data_trans import pairs_to_array

class Rankings:
    def __init__(self, theorems, theorems_features, chronology,
                 model, params_data_trans, n_jobs=-1):
        """params_data_trans must contain list_of_features"""
        # be careful: backend 'loky' is needed to not colide with model
        # 'loky' is available only in the newest dev release of joblib
        # (only on github so far)
        with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
            drfm = delayed(rfm)
            rankings = parallel(drfm(theorem, theorems_features,
                                set(chronology[:chronology.index(theorem)]),
                                model, params_data_trans)
                for theorem in theorems)
        self.rankings = dict(rankings)

    def rfm(theorem, theorems_features, available_premises,
            model, params_data_trans):
        """wrapper for ranking_from_model() needed for parallelization"""
        return (theorem, ranking_from_model(theorem, theorems_features,
                       available_premises, model, params_data_trans))

    def ranking_from_model(theorem, theorems_features, available_premises,
                            params_data_trans):
        pairs = [(theorems_features[theorem], theorems_features[premise])
                 for premise in available_premises]
        scores = score_pairs(pairs, model, params_data_trans)
        premises_scores = list(zip(available_premises, scores))
        premises_scores.sort(key = lambda x: x[1], reverse = True)
        return premises_scores

    def score_pairs(pairs, model, params_data_trans):
        array = pairs_to_array(pairs, params_data_trans)
        return model.predict(array) #TODO remember -- DMatrix...

    def __len__(self):
        return len(self.rankings)

    def __getitem__(self, theorem):
        return self.rankings[theorem]

    def add(self, theorem, ranking):
        self.rankings[theorem] = ranking


