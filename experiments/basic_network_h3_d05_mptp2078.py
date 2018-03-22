import sys, os
from random import sample
sys.path.append('..')
import premises as prs


N_JOBS = 15
DATA_DIR = 'data/MPTP2078'
OUTPUT_DIR = __file__.strip('.py')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
LOG_FILE = __file__.replace('.py', '.log')
ATP_DIR = os.path.join(OUTPUT_DIR, 'atp')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')


statements = prs.Statements(from_file=os.path.join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=os.path.join(DATA_DIR, 'features'),
                        logfile=LOG_FILE)
chronology = prs.Chronology(from_file=os.path.join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
proofs_train = prs.Proofs(from_file=os.path.join(DATA_DIR, 'atpproved.train'),
                          logfile=LOG_FILE)
proofs_test = prs.Proofs(from_file=os.path.join(DATA_DIR, 'atpproved.test'),
                          logfile=LOG_FILE)
train_theorems = set(proofs_train)
test_theorems = set(proofs_test)

params_data_trans = {'features': features,
                     'num_of_features': 0.2,
                     'chronology': chronology,
                     'ratio_neg_pos': 10,
                     'merge_mode': 'concat'}
train_labels, train_array = prs.proofs_to_train(proofs_train, params_data_trans,
                                               n_jobs=N_JOBS, logfile=LOG_FILE)
test_labels, test_array = prs.proofs_to_train(proofs_test, params_data_trans,
                                               n_jobs=N_JOBS, logfile=LOG_FILE)

params_train = {'model': 'network',
                'activation': 'relu',
                'batch_size': 100,
                'learning_rate': 0.01,
                'epochs': 500,
                'layers': 3,
                'hidden_layer': 100,
                'dropout': 0.5,
                'num_of_features': train_array.shape[1]}

model_path = prs.train(train_labels, train_array, test_labels, test_array,
       params=params_train, model_dir=MODEL_DIR, n_jobs=N_JOBS,
                       logdir=LOG_DIR, logfile=LOG_FILE)

rankings_test = prs.Rankings(test_theorems, model_path, params_data_trans,
                     model_type=params_train['model'],
                             n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_train = prs.Rankings(train_theorems, model_path, params_data_trans,
                     model_type=params_train['model'],
                             n_jobs=N_JOBS, logfile=LOG_FILE)

params_atp_eval = {}
proofs_test = prs.atp_evaluation(rankings_test, statements,
     params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
prs.utils.printline("Statistics for proofs in test set.", logfile=LOG_FILE)
proofs_test.print_stats(logfile=LOG_FILE)

proofs_train = prs.atp_evaluation(rankings_train, statements,
     params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
prs.utils.printline("Statistics for proofs in training set.", logfile=LOG_FILE)
proofs_train.print_stats(logfile=LOG_FILE)
