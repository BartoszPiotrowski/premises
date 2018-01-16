import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

DATA_DIR = 'data/debug_data'
ATP_DIR = 'atp'
LOG_FILE = 'tests/test_basic.log'
N_JOBS = 4

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
                     'features_ordered': features.all_features(),
                     'chronology': chronology,
                     'sparse': False}
rankings_train = prs.knn(test_theorems, proofs_train, params_data_trans)
params_atp_eval = {}
proofs_test = prs.atp_evaluation(rankings_train, statements, params_atp_eval,
                                 dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)