from .utils import read_dict, remove_supersets, lines_from_txt
from .data_transformation import pairs_to_array
from joblib import Parallel, delayed

class Features:
    def __init__(self, from_dict={}, from_file=''):
        if from_file:
            self.features = read_dict(from_file, type_of_values=list)
        else:
            self.features = from_dict

    def __len__(self):
        return len(self.features)

    def __getitem__(self, theorem):
        return self.features[theorem]

    def add(self, theorem, features):
        self.features[theorem] = features

    def all_features(self):
        return list(set().union(*self.features.values()))

class Statements:
    def __init__(self, from_dict=None, from_file=''):
        if from_file:
            lines = lines_from_txt(from_file)
            names = [l.split(',')[0].replace('fof(', '').replace(' ', '')
                        for l in lines]
            self.statements = dict(zip(names, lines))
        elif from_dict:
            self.statements = from_dict
        else:
            print("Error: no dict or file name provided to initialize from.")

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, theorem):
        return self.statements[theorem]

    def add(self, theorem, statements):
        self.statements[theorem] = statements


class Chronology:
    def __init__(self, from_list=None, from_file=''):
        if from_file:
            self.chronology = lines_from_txt(from_file)
        elif from_list:
            self.chronology = from_list
        else:
            print("Error: no list or file name provided to initialize from.")

    def __len__(self):
        return len(self.chronology)

    def __getitem__(self, index):
        return self.chronology[index]

    def index(self, theorem):
        if theorem in set(self.chronology):
            return self.chronology.index(theorem)
        else:
            print("Error: theorem {} not contained in chronology list.".format(
                                theorem))

    def available_premises(self, theorem):
        if theorem in set(self.chronology):
            return self.chronology[:self.index(theorem)]
        else:
            print("Error: theorem {} not contained in chronology list.".format(
                                theorem))

class Proofs:
    def __init__(self, from_dict={}, from_file=''):
        if from_file:
            prfs = read_dict(from_file, type_of_values=list, sep_in_list=' ')
            self.proofs = {thm: [set(prfs[thm])] for thm in prfs}
        else:
            self.proofs = from_dict

    def __len__(self):
        return len(self.proofs)

    def __getitem__(self, theorem):
        return self.proofs[theorem]

    def __iter__(self):
        return self.proofs.__iter__()

    def add(self, theorem, proof):
        proof = set(proof)
        if not theorem in self.proofs:
            self.proofs[theorem] = [proof]
        else:
            for prf in self.proofs[theorem]:
                if proof >= prf:
                    break
                if proof < prf:
                    prf &= proof
                    break
            else:
                self.proofs[theorem].append(proof)
            self.proofs[theorem] = remove_supersets(self.proofs[theorem])

    def update(self, new_proofs):
        for thm in new_proofs:
            for prf in new_proofs[thm]:
                self.add(thm, prf)

    def nums_of_proofs(self):
        return [len(self[t]) for t in self.proofs]

    def num_of_all_proofs(self):
        return sum(self.nums_of_proofs())

    def avg_num_of_proofs(self):
        return self.num_of_all_proofs() / len(self)

class Rankings:
    def __init__(self, theorems, model, params, n_jobs=-1):
        """params must contain list_of_features"""
        # be careful: backend 'loky' is needed to not colide with model
        # 'loky' is available only in the newest dev release of joblib
        # (only on github so far)
        with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
            drfm = delayed(rfm)
            rankings = parallel(drfm(theorem, model, params)
                for theorem in theorems)
        self.rankings = dict(rankings)

    def rfm(theorem, model, params):
        """wrapper for ranking_from_model() needed for parallelization"""
        return (theorem, ranking_from_model(theorem, model, params))

    def ranking_from_model(theorem, model, params):
        features = params['features']
        chronology = params['chronology']
        available_premises = chronology.available_premises(theorem)
        pairs = [(features[theorem], features[premise])
                 for premise in available_premises]
        scores = score_pairs(pairs, model, params)
        premises_scores = list(zip(available_premises, scores))
        premises_scores.sort(key = lambda x: x[1], reverse = True)
        return premises_scores

    def score_pairs(pairs, model, params):
        array = pairs_to_array(pairs, params)
        return model.predict(array) #TODO remember -- DMatrix...

    def __len__(self):
        return len(self.rankings)

    def __getitem__(self, theorem):
        return self.rankings[theorem]

    def add(self, theorem, ranking):
        self.rankings[theorem] = ranking




if __name__ == "__main__":
    prfs = Proofs()
    prfs.add("t1", ["p3", "p2"])
    prfs.add("t1", ["p5", "p2"])
    prfs.add("t2", ["p1"])
    prfs.add("t1", ["p4"])
    prfs.add("t1", ["p2"])
    print(prfs.num_of_all_proofs())
    print(prfs.nums_of_proofs())
    print(prfs["t1"])

