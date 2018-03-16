import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 50
DATA_DIR = 'data/mizar40'
ATP_DIR = 'atp'
RANKINGS_DIR = 'rankings_knn_multi'
LOG_FILE = __file__.replace('.py', '.log')

statements = prs.Statements(from_file=join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
proofs_train = prs.Proofs(from_file=join(DATA_DIR, 'atpproved_multi'),
                          logfile=LOG_FILE)
proofs_train.print_stats(logfile=LOG_FILE)
train_theorems = set(proofs_train)
test_theorems = prs.utils.readlines(join(DATA_DIR, 'theorems_not_atpproved'))
assert not set(test_theorems) & set(train_theorems)
params = {'features': features,
          'chronology': chronology,
          'N': 100,
          'rankings_dir': RANKINGS_DIR}
rankings_test = prs.knn(test_theorems, proofs_train, params, n_jobs=N_JOBS)
