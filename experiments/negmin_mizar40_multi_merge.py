import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 52
DATA_DIR = 'data/mizar40'
ATP_DIR = 'atp'
MODELS_DIR = 'models'
RANKINGS_DIR = 'rankings_negmin_mizar40'
LOG_FILE = __file__.replace('.py', '.log')

statements = prs.Statements(from_file=join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
proofs_train = prs.Proofs(from_file=join(DATA_DIR, 'atpproved_multi'),
                          logfile=LOG_FILE)
proofs_train.print_stats(logfile=LOG_FILE)
train_theorems = set(proofs_train)
test_theorems = prs.utils.readlines(join(DATA_DIR, 'theorems_not_atpproved'))
assert not set(test_theorems) & set(train_theorems)
params_data_trans = {'features': features,
                     'num_of_features': 0.3,
                     'chronology': chronology,
                     'ratio_neg_pos': 20}
train_labels, train_array = prs.proofs_to_train(proofs_train,
                    params_data_trans, n_jobs=N_JOBS, logfile=LOG_FILE)
params_train = {'num_boost_round': 2000}
params_train = {'eta': 0.3}
model = prs.train(train_labels, train_array, params=params_train,
                    n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_train = prs.Rankings(train_theorems, model, params_data_trans,
                     n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_test = prs.Rankings(test_theorems, model, params_data_trans,
                     n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_test.save_rankings_to_dir(RANKINGS_DIR)
params_atp_eval = {}
proofs_train.update(prs.atp_evaluation(rankings_train, statements,
           params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE))
prs.utils.printline("STATS OF TRAINING PROOFS", logfile=LOG_FILE)
proofs_train.print_stats(logfile=LOG_FILE)
params_data_trans['level_of_negative_mining'] = 'random'
for i in range(40):
    prs.utils.printline("NEGATIVE MINING ROUND: {}".format(i + 1),
                        logfile=LOG_FILE)
    params_data_trans['rankings_for_negative_mining'] = rankings_train
    train_labels, train_array = prs.proofs_to_train(proofs_train,
                    params_data_trans, n_jobs=N_JOBS, logfile=LOG_FILE)
    model = prs.train(train_labels, train_array, params=params_train,
                        n_jobs=N_JOBS, logfile=LOG_FILE)
    rankings_train = prs.Rankings(train_theorems, model, params_data_trans,
                         to_merge=rankings_train, n_jobs=N_JOBS, logfile=LOG_FILE)
    rankings_test = prs.Rankings(test_theorems, model, params_data_trans,
                         to_merge=rankings_test, n_jobs=N_JOBS, logfile=LOG_FILE)
    rankings_test.save_rankings_to_dir(RANKINGS_DIR)
    proofs_train.update(prs.atp_evaluation(rankings_train, statements,
          params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE))
    prs.utils.printline("STATS OF TRAINING PROOFS", logfile=LOG_FILE)
    proofs_train.print_stats(logfile=LOG_FILE)
