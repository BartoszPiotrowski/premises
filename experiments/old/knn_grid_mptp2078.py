import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 25
DATA_DIR = 'data/MPTP2078'
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
params = {'features': features,
          'features_ordered': features.all_features(),
          'chronology': chronology}
Ns = [10, 20, 50, 100, 200, 500]
for N in Ns:
    prs.utils.printline("Neighbours: {}".format(N), logfile=LOG_FILE)
    params['N'] = N
    rankings_train = prs.knn(test_theorems, proofs_train, params)
    params_atp_eval = {}
    proofs_test = prs.atp_evaluation(rankings_train, statements, params_atp_eval,
                             dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
    proofs_test.print_stats(logfile=LOG_FILE)