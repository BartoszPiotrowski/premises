from .utils import read_dict, remove_supersets, readlines, printline, shuffled
from .data_transformation import pairs_to_array
from joblib import Parallel, delayed
import xgboost
from time import time

class Features:
    def __init__(self, from_dict={}, from_file='', verbose=True, logfile=''):
        if from_file:
            self.features = read_dict(from_file, type_of_values=list)
        else:
            self.features = from_dict
        if verbose or logfile:
            message = "Features of {} theorems and definitions loaded.".format(
                       len(self))
            printline(message, logfile, verbose)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, theorem):
        return self.features[theorem]

    def __contains__(self, theorem):
        return theorem in self.features

    def add(self, theorem, features):
        self.features[theorem] = features

    def all_features(self):
        return list(set().union(*self.features.values()))

class Statements:
    def __init__(self, from_dict=None, from_file='', verbose=True, logfile=''):
        if from_file:
            lines = readlines(from_file)
            names = [l.split(',')[0].replace('fof(', '').replace(' ', '')
                        for l in lines]
            self.statements = dict(zip(names, lines))
        elif from_dict:
            self.statements = from_dict
        else:
            print("Error: no dict or file name provided to initialize from.")
        if verbose or logfile:
            message = "Statements of {} theorems and definitions loaded.".format(
                len(self))
            printline(message, logfile, verbose)

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, theorem):
        return self.statements[theorem]

    def __contains__(self, theorem):
        return theorem in self.statements

    def add(self, theorem, statements):
        self.statements[theorem] = statements


class Chronology:
    def __init__(self, from_list=None, from_file='', verbose=True, logfile=''):
        if from_file:
            self.chronology = readlines(from_file)
        elif from_list:
            self.chronology = from_list
        else:
            print("Error: no list or file name provided to initialize from.")
        if verbose or logfile:
            message = ("Chronological order of {} theorems "
                       "and definitions loaded.").format( len(self))
            printline(message, logfile, verbose)

    def __len__(self):
        return len(self.chronology)

    def __getitem__(self, index):
        return self.chronology[index]

    def __contains__(self, theorem):
        return theorem in set(self.chronology)

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
    def __init__(self, from_dict={}, from_file='', verbose=True, logfile=''):
        if from_file:
            prfs = read_dict(from_file, type_of_values=list, sep_in_list=' ')
            self.proofs = {thm: [set(prfs[thm])] for thm in prfs}
        else:
            self.proofs = {}
            self.update(from_dict)
        if verbose or logfile:
            message = "Proofs of {} theorems loaded.".format(
                len(self))
            printline(message, logfile, verbose)

    def __len__(self):
        return len(self.proofs)

    def __getitem__(self, theorem):
        return self.proofs[theorem]

    def __iter__(self):
        return self.proofs.__iter__()

    def __contains__(self, theorem):
        return theorem in self.proofs

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

    def update(self, new_proofs, verbose=True, logfile=''):
        for thm in new_proofs:
            for prf in new_proofs[thm]:
                self.add(thm, prf)

    def union_of_proofs(self, theorem):
        return set().union(*self.proofs[theorem])

    def nums_of_proofs(self):
        return [len(self[t]) for t in self.proofs]

    def hist_nums_of_proofs(self):
        d = {}
        ns = self.nums_of_proofs()
        for i in range(max(ns)):
            s = sum([n == i + 1 for n in ns])
            if s > 0:
                d[i + 1] = s
        return d

    def num_of_all_proofs(self):
        return sum(self.nums_of_proofs())

    def avg_num_of_proofs(self):
        return self.num_of_all_proofs() / len(self)

    def avg_length_of_proof(self):
        lengths = [len(p) for t in self.proofs for p in t]
        return lengths / len(lengths)

    def stats(self):
        return {'num_of_thms': len(self),
                'num_of_proofs': self.num_of_all_proofs(),
                'avg_num_of_proofs': self.avg_num_of_proofs(),
                'avg_len_of_proof': self.avg_num_of_proofs()}

    def print_stats(logfile=''):
        printline("Number of all theorems with proofs: {}".format(len(self)),
                  logfile)
        printline("Number of all proofs: {}".format(self.num_of_all_proofs),
                  logfile)
        printline("Average number of proofs per theorem: {}".format(
                  self.avg_num_of_proofs), logfile)
        ns = self.hist_nums_of_proofs()
        for n in ns:
            printline("Number of theorems with exactly {} proof(s): {}".format(
                n, ns[n]), logfile)
        printline("Average length of a proof: {}".format(
                  self.avg_length_of_proof), logfile)


class Rankings:
    def __init__(self, theorems, model=None, params=None, verbose=True,
                 logfile='', n_jobs=-1):
        assert 'chronology' in params

        if model:
            assert 'features' in params
            assert 'features_ordered' in params
            if verbose or logfile:
                message = ("Creating rankings of premises from the trained model "
                           "for {} theorems...").format(len(theorems))
                printline(message, logfile, verbose)
            # be careful: backend 'loky' is needed to not colide with model
            # 'loky' is available only in the newest dev release of joblib
            # (only on github so far)
            with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
                drfm = delayed(self.rfm)
                rankings_with_scores = parallel(drfm(theorem, model, params)
                                                for theorem in theorems)
            self.rankings_with_scores = dict(rankings_with_scores)
            self.rankings = self._rankings_only_names(self.rankings_with_scores)
        else:
            if verbose or logfile:
                message = ("Creating random rankings of premises "
                           "for {} theorems...").format(len(theorems))
                printline(message, logfile, verbose)
            chronology = params['chronology']
            random_rankings = {thm: shuffled(chronology.available_premises(thm))
                               for thm in theorems}
            self.rankings = random_rankings

        if verbose or logfile:
            message = "Rankings created."
            printline(message, logfile, verbose)

    def _rankings_only_names(self, rankings_with_scores):
        return {thm: [rankings_with_scores[thm][i][0]
                          for i in range(len(rankings_with_scores[thm]))]
                             for thm in rankings_with_scores}

    def rfm(self, theorem, model, params):
        """wrapper for ranking_from_model() needed for parallelization"""
        return (theorem, self.ranking_from_model(theorem, model, params))

    def ranking_from_model(self, theorem, model, params):
        features = params['features']
        chronology = params['chronology']
        available_premises = chronology.available_premises(theorem)
        pairs = [(features[theorem], features[premise])
                 for premise in available_premises]
        scores = self.score_pairs(pairs, model, params)
        premises_scores = list(zip(available_premises, scores))
        premises_scores.sort(key = lambda x: x[1], reverse = True)
        return premises_scores

    def score_pairs(self, pairs, model, params):
        array = pairs_to_array(pairs, params)
        if isinstance(model, xgboost.Booster):
            array = xgboost.DMatrix(array)
        return model.predict(array)

    def __len__(self):
        return len(self.rankings)

    def __getitem__(self, theorem):
        return self.rankings[theorem]

    def __iter__(self):
        return self.rankings.__iter__()

    def __contains__(self, theorem):
        return theorem in self.rankings

    def add(self, theorem, ranking):
        self.rankings[theorem] = ranking

