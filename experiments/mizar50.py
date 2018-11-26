import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 2
DATA_DIR = 'data/mizar40'
ATP_DIR = 'atp'
LOG_FILE = __file__.replace('.py', '.log')

statements = prs.Statements(from_file=join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
proofs_train = prs.Proofs(from_file=join(DATA_DIR, 'allproofs_small'),
                          logfile=LOG_FILE)

params_data_trans = {'features': features,
                     'chronology': chronology,
                     'ratio_neg_pos': 128}

train_labels, train_array = prs.proofs_to_train(proofs_train,
                    params_data_trans, n_jobs=N_JOBS, logfile=LOG_FILE)

params_train = {'num_boost_round': 80}

model = prs.train(train_labels, train_array, params=params_train,
                    n_jobs=N_JOBS, logfile=LOG_FILE)

model.save_model("model_xgboost_mizar50.pkl")
