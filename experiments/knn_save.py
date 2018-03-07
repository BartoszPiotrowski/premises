import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 35
DATA_DIR = 'data/MPTP2078'
ATP_DIR = 'atp'
RANKINGS_DIR = 'rankings_knn'
PROOFS_DIR = 'proofs_knn'
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
params['N'] = 100
rankings_test = prs.knn(test_theorems, proofs_train, params)
rankings_test.save_rankings_to_dir(RANKINGS_DIR, with_scores=True)
params_atp_eval = {}
proofs_test = prs.atp_evaluation(rankings_test, statements, params_atp_eval,
                         dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
proofs_test.print_stats(logfile=LOG_FILE)
proofs_test.save_atp_useful_to_dir(PROOFS_DIR)
