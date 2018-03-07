from .utils import read_dict, remove_supersets, readlines, printline
from .utils import shuffled, partition, writelines
from .data_transformation import pairs_to_array
import os
from joblib import Parallel, delayed
import xgboost
from random import sample
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
        self.num_of_features = len(self.order_of_features)

        if verbose or logfile:
            message = "Features of {} thms and definitions loaded.".format(
                       len(self))
            printline(message, logfile, verbose)

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self.features.__iter__()

    def __getitem__(self, thm):
        return self.features[thm]

    def __contains__(self, thm):
        return thm in self.features

    def add(self, thm, features):
        self.features[thm] = features

    def all_features(self):
        return list(set().union(*self.features.values()))

    def dict_features_thms(self):
        dict_features_thms = {}
        for thm in self:
            for f in self[thm]:
                try: dict_features_thms[f].add(thm)
                except: dict_features_thms[f] = {thm}
        return dict_features_thms

    def dict_features_numbers(self):
        dft = self.dict_features_thms()
        return {f:len(dft[f]) for f in dft}

    def subset(self, thms):
        return Features(from_dict={thm: self[thm] for thm in thms}, verbose=False)

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

    def __getitem__(self, thm):
        return self.statements[thm]

    def __contains__(self, thm):
        return thm in self.statements

    def add(self, thm, statements):
        self.statements[thm] = statements


class Chronology:
    def __init__(self, from_list=None, from_file='', verbose=True, logfile=''):
        if from_file:
            self.chronology = readlines(from_file)
        elif from_list:
            self.chronology = from_list
        else:
            print("Error: no list or file name provided to initialize from.")
        if verbose or logfile:
            message = ("Chronological order of {} thms "
                       "and definitions loaded.").format( len(self))
            printline(message, logfile, verbose)

    def __len__(self):
        return len(self.chronology)

    def __getitem__(self, index):
        return self.chronology[index]

    def __contains__(self, thm):
        return thm in set(self.chronology)

    def index(self, thm):
        if thm in set(self.chronology):
            return self.chronology.index(thm)
        else:
            print("Error: theorem {} not contained in chronology list.".format(
                                thm))

    def available_premises(self, thm):
        if thm in self.chronology:
            return self.chronology[:self.index(thm)]
        else:
            print("Error: theorem {} not contained in chronology list.".format(
                                thm))
    def initial_segment(self, i):
        return self.chronology[:i]

class Proofs:
    def __init__(self, from_dict={}, from_file='', verbose=True, logfile=''):
        if from_file:
            '''
            The file with 'proofs' is supposed to contain a list of theorems'
            names, each theorem associated a list of premises' names used in its
            proof -- in the following form:
            thm1: prm1 prm2 prm3
            thm2: prm4 prm5
            .
            .
            .
            '''
            prfs_dict = self.read_proofs(from_file)
        else:
            '''
            The dict with 'proofs' should have theorems' names as keys and each
            value should be a list containing sets of premises' names used in
            different proofs of the given theorem.
            '''
            prfs_dict = from_dict
        self.proofs = {}
        self.update(prfs_dict)
        if verbose or logfile:
            message = "Proofs of {} theorems loaded.".format(len(self))
            printline(message, logfile, verbose)

    def read_proofs(self, filename, sep=':', sep_in_list=' '):
        with open(filename) as file:
            slines = [l.split(sep) for l in file.read().splitlines()]
        names = [l[0] for l in slines]
        proofs_raw = [l[1] for l in slines]
        proofs = []
        for v in proofs_raw:
            proofs.append([i for i in v.split(sep_in_list)])
        proofs_dict = {}
        for i in range(len(names)):
            try: proofs_dict[names[i]].append(set(proofs[i]))
            except: proofs_dict[names[i]] = [set(proofs[i])]
        return proofs_dict

    def __len__(self):
        return len(self.proofs)

    def __getitem__(self, thm):
        return self.proofs[thm]

    def __iter__(self):
        return self.proofs.__iter__()

    def __contains__(self, thm):
        return thm in self.proofs

    def add(self, thm, proof):
        proof = set(proof)
        if not thm in self.proofs:
            self.proofs[thm] = [proof]
        else:
            for prf in self.proofs[thm]:
                if proof >= prf:
                    break
                if proof < prf:
                    prf &= proof
                    break
            else:
                self.proofs[thm].append(proof)
            self.proofs[thm] = remove_supersets(self.proofs[thm])

    def update(self, new_proofs, verbose=True, logfile=''):
        for thm in new_proofs:
            assert isinstance(new_proofs[thm], list)
            if len(new_proofs[thm]) > 0:
                for prf in new_proofs[thm]:
                    if len(prf) > 0:
                        self.add(thm, prf)

    def subset(self, thms):
        prfs_dict = {thm: self[thm] for thm in thms}
        return Proofs(from_dict=prfs_dict)

    def random_subset(self, n):
        assert n > 0
        if n < 1:
            n = int(n * len(self))
        thms_sample = sample(set(self), n)
        return self.subset(thms_sample)

    def union_of_proofs(self, thm):
        return set().union(*self.proofs[thm])

    def union_of_short_proofs(self, thm):
        min_length = min([len(p) for p in self[thm]])
        short_proofs = [p for p in self[thm] if len(p) <= min_length + 1]
        return set().union(*short_proofs)

    def unions_of_proofs(self):
        return {thm: self.union_of_proofs(thm) for thm in self}

    def with_trivial(self, thms_properties=None):
        with_trivial = {thm: self[thm] + [{thm}] for thm in self}
        if thms_properties:
            only_trivial = {thm_prt: [{thm_prt}]
                            for thm_prt in set(thms_properties) - set(self)}
        else:
            only_trivial = {}
        return {**with_trivial, **only_trivial}

    def dict_premises_thms(self):
        dict_premises_thms = {}
        for thm in self:
            for prm in self[thm]:
                try: dict_premises_thms[prm].add(thm)
                except: dict_premises_thms[prm] = {thm}
        return dict_premises_thms

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

    def thms_with_max_number_of_proofs(self):
        maxim = max(self.nums_of_proofs())
        return [thm for thm in self.proofs if len(self[thm]) == maxim]

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
        printline("Average number of premises used in one proof: {:.3f}".format(
                  self.avg_length_of_proof()), logfile)
        printline("Theorems with maximal number of proofs found: {}".format(
                   self.thms_with_max_number_of_proofs()), logfile)
        #thm_max = self.thms_with_max_number_of_proofs()[0]
        #printline("Distribution of lengths of proofs for theorem {}: {}".format(
        #                 thm_max, [len(p) for p in self[thm_max]]), logfile)
        #for p in self[thm_max]:
        #    print(p)

    def save_proofs_to_file(self, file_name):
        lines = ['{}:{}'.format(thm, ' '.join(list(prf)))
                 for thm in self for prf in self[thm]]
        writelines(lines, file_name)

    def save_atp_useful_to_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for thm in self:
            atp_useful = set().union(*self[thm])
            writelines(atp_useful, os.path.join(dir_name, thm))

class Rankings:
    def __init__(self, thms=None, model=None, params=None, from_dict=None,
                 to_merge=None, merge_mode='geom', save_to_dir=None,
                 verbose=True, logfile='', n_jobs=-1):
        if save_to_dir:
            if not os.path.exists(save_to_dir):
                os.mkdir(save_to_dir)
        if from_dict:
            ex = from_dict[list(from_dict)[0]]
            if type(ex) == str:
                self.rankings = from_dict
                self.rankings_with_scores = None
            else:
                self.rankings_with_scores = from_dict
                self.rankings = self._rankings_only_names(from_dict)
        elif model:
            time0 = time()
            assert 'chronology' in params
            assert 'features' in params
            chronology = params['chronology']
            features = params['features']
            params_small = {'merge_mode': params['merge_mode'],
                            'num_of_features': params['num_of_features']}
            print("BEFORE", len(thms))
            thms = [t for t in thms if len(chronology.available_premises(t))]
            if verbose or logfile:
                message = ("Creating rankings of premises from the trained model "
                           "for {} theorems...").format(len(thms))
                printline(message, logfile, verbose)
            print("AFTER", len(thms))
            # be careful: backend 'loky' is needed to not colide with model
            # 'loky' is available only in the newest dev release of joblib
            # (only on github so far)
            with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
                drfm = delayed(self.ranking_from_model)
                rankings_with_scores = parallel(drfm(
                 thm, model,
                 chronology.available_premises(thm),
                 features.subset(set(chronology.available_premises(thm)) | {thm}),
                 params_small, save_to_dir) for thm in thms)
            rankings_with_scores = dict(rankings_with_scores)
            rankings = self._rankings_only_names(rankings_with_scores)
            if to_merge:
                assert set(thms) == set(to_merge)
                merged = self.merge_rankings(rankings, to_merge, mode=merge_mode)
                self.rankings = merged
                self.rankings_with_scores = None
            else:
                self.rankings = rankings
                self.rankings_with_scores = rankings_with_scores
            print("ALL:", time()-time0)
        else:
            if verbose or logfile:
                message = ("Creating random rankings of premises "
                           "for {} theorems...").format(len(thms))
                printline(message, logfile, verbose)
            chronology = params['chronology']
            random_rankings = {thm: shuffled(chronology.available_premises(thm))
                               for thm in thms}
            self.rankings = random_rankings
            self.rankings_with_scores = None

        if verbose or logfile:
            message = "Rankings created."
            printline(message, logfile, verbose)

    def _rankings_only_names(self, rankings_with_scores):
        return {thm: [rankings_with_scores[thm][i][0]
                          for i in range(len(rankings_with_scores[thm]))]
                             for thm in rankings_with_scores}

    def ranking_from_model(self, thm, model, available_premises, features,
                           params, save_to_dir=None):
        time0 = time()
        assert len(available_premises)
        features_thm = features[thm]
        pairs = [(features_thm, features[prm])
                 for prm in available_premises]
        #time1=time(); print("1", time1-time0)
        scores = self.score_pairs(pairs, model, params)
        #time2=time(); print("2", time2-time1)
        premises_scores = list(zip(available_premises, scores))
        #time3=time(); print("3", time3-time2)
        premises_scores.sort(key = lambda x: x[1], reverse = True)
        #time4=time(); print("4", time4-time3)
        print(thm, time()-time0)
        # rankings cut to 1024 for efficiency when parallelizing
        ranking_with_scores = premises_scores[:1024]
        if save_to_dir:
            ranking = [t[0] for t in ranking_with_scores]
            writelines(ranking, os.path.join(save_to_dir, thm))
        return (thm, ranking_with_scores)

    def score_pairs(self, pairs, model, params):
        assert len(pairs)
        time0 = time()
        array = pairs_to_array(pairs, params)
        time1=time(); print("1", time1-time0)
        if isinstance(model, xgboost.Booster):
            array = xgboost.DMatrix(array)
        time2=time(); print("2", time2-time1)
        return model.predict(array)

    def merge_two_rankings(self, ranking1, ranking2, mode='geom'):
        comb = {
            'aryt': lambda x, y: (x + y) / 2,
            'geom': lambda x, y: (x * y)**.5
        }[mode]
        scores_ranking1 = {ranking1[i]: i for i in range(len(ranking1))}
        scores_ranking2 = {ranking2[i]: i for i in range(len(ranking2))}
        scores_ranking = {p: comb(scores_ranking1[p], scores_ranking2[p]) \
                        if p in set(scores_ranking1) & set(scores_ranking2) else \
                           (scores_ranking1[p] if p in scores_ranking1 else \
                            scores_ranking2[p]) \
                            for p in set(scores_ranking1) | set(scores_ranking2)}
        ranking = sorted(scores_ranking.items(), key=lambda x: x[1])
        return [i[0] for i in ranking]

    def merge_rankings(self, rankings_a, rankings_b, mode="geom"):
        assert set(rankings_a) == set(rankings_b)
        return {thm: self.merge_two_rankings(rankings_a[thm], rankings_b[thm],
                                               mode=mode) for thm in rankings_a}

    def save_ranking_to_file(self, thm, file_name, with_scores=False):
        if with_scores:
            writelines(self.rankings_with_scores[thm], file_name)
        else:
            writelines(self[thm], file_name)

    def save_rankings_to_dir(self, dir_name, with_scores=False):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for thm in self:
            self.save_ranking_to_file(thm, os.path.join(dir_name, thm),
                                      with_scores)

    def __len__(self):
        return len(self.rankings)

    def __getitem__(self, thm):
        return self.rankings[thm]

    def __iter__(self):
        return self.rankings.__iter__()

    def __contains__(self, thm):
        return thm in self.rankings

    def add(self, thm, ranking):
        self.rankings[thm] = ranking

