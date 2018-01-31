import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 50
DATA_DIR = 'data/mizar40'
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
                     'num_of_features': 0.3,
                     'chronology': chronology,
                     'merge_mode': 'comb',
                     'ratio_neg_pos': 28}
train_labels, train_array = prs.proofs_to_train(proofs_train,
                    params_data_trans, n_jobs=N_JOBS, logfile=LOG_FILE)
params_train = {'num_boost_round': 5000}
params_train = {'eta': 0.2}
model = prs.train(train_labels, train_array, params=params_train,
                    n_jobs=N_JOBS, logfile=LOG_FILE)
model.save_model("model_xgboost_mizar40.pkl")
rankings_test = prs.Rankings(test_theorems, model, params_data_trans,
                     n_jobs=N_JOBS, logfile=LOG_FILE)

params_atp_eval = {}
proofs_test = prs.atp_evaluation(rankings_test, statements, params_atp_eval,
                         dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
proofs_test.print_stats(logfile=LOG_FILE)
