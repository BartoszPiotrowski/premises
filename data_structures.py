from .utils import read_dict, remove_supersets, readlines, printline, shuffled
from .data_transformation import pairs_to_array
from joblib import Parallel, delayed
import xgboost
from time import time

class Features:
    def __init__(self, from_dict={}, from_file='', verbose=True, logfile=''):
        if from_file:
            f_dict = read_dict(from_file, type_of_values=list)
        elif from_dict:
            f_dict = from_dict
        else:
            print("Error: provide file or dictionary with features.")
        self.features = {f: set(f_dict[f]) for f in f_dict}
        self.order_of_features = self.all_features()

        if verbose or logfile:
            message = "Features of {} theorems and definitions loaded.".format(
                       len(self))
            printline(message, logfile, verbose)

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self.features.__iter__()

    def __getitem__(self, theorem):
        return self.features[theorem]

    def __contains__(self, theorem):
        return theorem in self.features

    def add(self, theorem, features):
        self.features[theorem] = features

    def all_features(self):
        return list(set().union(*self.features.values()))

    def dict_features_theorems(self):
        dict_features_theorems = {}
        for thm in self:
            for f in self[thm]:
                try: dict_features_theorems[f].add(thm)
                except: dict_features_theorems[f] = {thm}
        return dict_features_theorems

    def dict_features_numbers(self):
        dft = self.dict_features_theorems()
        return {f:len(dft[f]) for f in dft}

    def subset(self, thms):
        return Features(from_dict={thm: self[thm] for thm in thms})

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

    def __iter__(self):
        return self.statements.__iter__()

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
            message = "Proofs of {} theorems loaded.".format(len(self))
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

    def unions_of_proofs(self):
        return {thm: self.union_of_proofs(thm) for thm in self}

    def with_trivial(self, theorems_properties=None):
        with_trivial = {thm: self[thm] + [{thm}] for thm in self}
        if theorems_properties:
            only_trivial = {thm_prt: [{thm_prt}]
                            for thm_prt in set(theorems_properties) - set(self)}
        else:
            only_trivial = {}
        return {**with_trivial, **only_trivial}

    def dict_premises_theorems(self):
        dict_premises_theorems = {}
        for thm in self:
            for prm in self[thm]:
                try: dict_premises_theorems[prm].add(thm)
                except: dict_premises_theorems[prm] = {thm}
        return dict_premises_theorems

    def nums_of_proofs(self):
        return [len(self[t]) for t in self]

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
        lengths = [len(p) for t in self.proofs for p in self.proofs[t]]
        return sum(lengths) / len(lengths)

    def stats(self):
        return {'num_of_thms': len(self),
                'num_of_proofs': self.num_of_all_proofs(),
                'avg_num_of_proofs': self.avg_num_of_proofs(),
                'avg_len_of_proof': self.avg_length_of_proof()}

    def print_stats(self, logfile=''):
        printline("Number of all theorems with proof(s): {}".format(len(self)),
                  logfile)
        printline("Number of all proofs: {}".format(self.num_of_all_proofs()),
                  logfile)
        ns = self.hist_nums_of_proofs()
        for n in ns:
            printline("Number of theorems with exactly {} proof(s): {}".format(
                n, ns[n]), logfile)
        printline("Average number of proofs per theorem: {:.3f}".format(
                  self.avg_num_of_proofs()), logfile)
        #printline("Average number of premises used in a proof: {:.3f}".format(
        #          self.avg_length_of_proof()), logfile)


class Rankings:
    def __init__(self, theorems=None, model=None, params=None, from_dict=None,
                 verbose=True, logfile='', n_jobs=-1):
        if from_dict:
            self.rankings_with_scores = from_dict
            self.rankings = self._rankings_only_names(self.rankings_with_scores)
        elif model:
            assert 'chronology' in params
            assert 'features' in params
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

