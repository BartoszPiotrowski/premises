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
proofs_test = prs.Proofs(from_file=join(DATA_DIR, 'atpproved.test'),
                          verbose=False)
test_theorems = set(proofs_test)
params_data_trans = {'features': features,
                     'chronology': chronology,
                     'merge_mode': 'concat',
                     'sparse': False}
train_labels, train_array = prs.proofs_to_train(proofs_train, params_data_trans,
                                               n_jobs=N_JOBS, logfile=LOG_FILE)
params_train = {'booster': 'gblinear'}
model = prs.train(train_labels, train_array, params=params_train,
                    n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_train = prs.Rankings(test_theorems, model, params_data_trans,
                     n_jobs=N_JOBS, logfile=LOG_FILE)
params_atp_eval = {}
proofs_test = prs.atp_evaluation(rankings_train, statements, params_atp_eval,
                                 dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
