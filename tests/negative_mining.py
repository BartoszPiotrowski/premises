import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = -1
DATA_DIR = 'data/debug_data'
ATP_DIR = 'atp'
LOG_FILE = __file__.replace('.py', '.log')

statements = prs.Statements(from_file=join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
proofs_train = prs.Proofs(from_file=join(DATA_DIR, 'atpproved.train'),
                          logfile=LOG_FILE)
theorems = prs.utils.readlines(join(DATA_DIR, 'theorems_atpproved'))
train_theorems = set(proofs_train)
test_theorems = set(theorems) - set(train_theorems)
params_data_trans = {'features': features,
                     'chronology': chronology,
                     'num_of_features': 0.1,
                     'merge_mode': 'concat'}
train_labels, train_array = prs.proofs_to_train(proofs_train, params_data_trans,
                                               n_jobs=N_JOBS, logfile=LOG_FILE)
params_train = {'num_boost_round': 10}
model = prs.train(train_labels, train_array, params=params_train,
                    n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_train = prs.Rankings(train_theorems, model, params_data_trans,
                     n_jobs=N_JOBS, logfile=LOG_FILE)
params_data_trans['rankings_for_negative_mining'] = rankings_train
params_data_trans['level_of_negative_mining'] = 'all'
train_labels, train_array = prs.proofs_to_train(proofs_train, params_data_trans,
                       n_jobs=N_JOBS, logfile=LOG_FILE)
model = prs.train(train_labels, train_array, params=params_train,
                    n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_test = prs.Rankings(test_theorems, model, params_data_trans,
                     n_jobs=N_JOBS, logfile=LOG_FILE)
params_atp_eval = {}
proofs_test = prs.atp_evaluation(rankings_test, statements, params_atp_eval,
                                 dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
