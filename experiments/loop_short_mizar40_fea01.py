import sys
from os.path import join
import random
random.seed(1)
sys.path.append('..')
import premises as prs

DATA_DIR = 'data/mizar40'
ATP_DIR = 'atp'
LOG_FILE = __file__.replace('.py', '.log')
N_JOBS = 50

statements = prs.Statements(from_file=join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
theorems = prs.utils.readlines(join(DATA_DIR, 'theorems_atpproved'))
params_data_trans = {'features': features,
                     'num_of_features': 0.1,
                     'chronology': chronology,
                     'only_short_proofs': True}

theorems_init_segment = [t for t in chronology.initial_segment(2000) \
                         if t in set(theorems)]

# randomly generated rankings
rankings_random = prs.Rankings(theorems_init_segment, model=None,
                   params=params_data_trans, n_jobs=N_JOBS, logfile=LOG_FILE)

proofs = prs.atp_evaluation(rankings_random, statements, dirpath=ATP_DIR,
                                 n_jobs=N_JOBS, logfile=LOG_FILE)

for i in range(40):
    prs.utils.printline("ITERATION: {}".format(i), LOG_FILE)

    theorems_init_segment = [t for t in chronology.initial_segment((i+2)*2000) \
                             if t in set(theorems)]
    prs.utils.printline("Number of theorems under consideration: {}".format(
                                len(theorems_init_segment)), LOG_FILE)
    train_labels, train_array = prs.proofs_to_train(proofs, params_data_trans,
                                           n_jobs=N_JOBS, logfile=LOG_FILE)
    params_train = {}
    model = prs.train(train_labels, train_array, params=params_train,
                        n_jobs=N_JOBS, logfile=LOG_FILE)
    rankings = prs.Rankings(theorems_init_segment, model, params_data_trans,
                         n_jobs=N_JOBS, logfile=LOG_FILE)
    params_atp_eval = {}
    proofs.update(prs.atp_evaluation(rankings, statements, params_atp_eval,
                         dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE))
    proofs.print_stats(logfile=LOG_FILE)
