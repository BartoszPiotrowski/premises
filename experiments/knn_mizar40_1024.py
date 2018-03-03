import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 25
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
test_theorems = set(theorems) - set(proofs_train)
params = {'features': features,
          'chronology': chronology}
Ns = [80, 160]
for N in Ns:
    prs.utils.printline("Neighbours: {}".format(N), logfile=LOG_FILE)
    params['N'] = N
    rankings_train = prs.knn(test_theorems, proofs_train, params)
    params_atp_eval = {'n_premises':[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]}
    proofs_test = prs.atp_evaluation(rankings_train, statements, params_atp_eval,
                             dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
    proofs_test.print_stats(logfile=LOG_FILE)
