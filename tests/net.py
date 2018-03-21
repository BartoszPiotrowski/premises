import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


N_JOBS = 1
DATA_DIR = 'data/debug_data'
ATP_DIR = 'atp'
LOG_DIR_NET = 'log_net'
MODEL_DIR = 'models'
LOG_FILE = __file__.replace('.py', '.log')

statements = prs.Statements(from_file=join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
proofs_train = prs.Proofs(from_file=join(DATA_DIR, 'atpproved.train'),
                          logfile=LOG_FILE)
proofs_test = prs.Proofs(from_file=join(DATA_DIR, 'atpproved.test'),
                          logfile=LOG_FILE)
train_theorems = set(proofs_train)
test_theorems = set(proofs_test)
params_data_trans = {'features': features,
                     'chronology': chronology,
                     'merge_mode': 'concat'}
train_labels, train_array = prs.proofs_to_train(proofs_train, params_data_trans,
                                               n_jobs=N_JOBS, logfile=LOG_FILE)
params_train_net = {'activation': 'relu',
                    'batch_size': 100,
                    'epochs': 10,
                    'layers': 2,
                    'hidden_layer': 100,
                    'logdir': LOG_DIR_NET,
                    'model_dir': MODEL_DIR,
                    'num_of_features': train_array.shape[1]}

model_path = prs.train_net(train_labels, train_array, params=params_train_net,
                    n_jobs=N_JOBS, logfile=LOG_FILE)

rankings_test = prs.Rankings(test_theorems, model_path, params_data_trans,
                     model_type='net', n_jobs=N_JOBS, logfile=LOG_FILE)

params_atp_eval = {}
proofs_test = prs.atp_evaluation(rankings_test, statements,
     params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
prs.utils.printline("STATS OF TEST PROOFS", logfile=LOG_FILE)
proofs_test.print_stats(logfile=LOG_FILE)
