import sys
from os.path import join
sys.path.append('..')
import premises as prs

DATA_DIR = 'data/MPTP2078'
ATP_DIR = 'atp'
LOG_FILE = __file__.replace('.py', '.log')
N_JOBS = -1 # all cores

statements = prs.Statements(from_file=join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
theorems = prs.utils.readlines(join(DATA_DIR, 'theorems'))
params_data_trans = {'features': features,
         'features_ordered': features.all_features(),
         'chronology': chronology,
         'sparse': False}

# randomly generated rankings
rankings_random = prs.Rankings(theorems, model=None, params=params_data_trans,
                             n_jobs=N_JOBS, logfile=LOG_FILE)

proofs = prs.atp_evaluation(rankings_random, statements, dirpath=ATP_DIR,
                                 n_jobs=N_JOBS, logfile=LOG_FILE)

for i in range(20):
    prs.utils.printline("ITERATION: {}".format(i), LOG_FILE)
    train_labels, train_array = prs.proofs_to_train(proofs, params_data_trans,
                                           n_jobs=N_JOBS, logfile=LOG_FILE)
    params_train = {}
    model = prs.train(train_labels, train_array, params=params_train,
                        n_jobs=N_JOBS, logfile=LOG_FILE)

    rankings = prs.Rankings(theorems, model, params_data_trans,
                         n_jobs=N_JOBS, logfile=LOG_FILE)
    params_atp_eval = {}
    proofs = prs.atp_evaluation(rankings, statements, params_atp_eval,
                             dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
    proofs.print_stats(logfile=LOG_FILE)