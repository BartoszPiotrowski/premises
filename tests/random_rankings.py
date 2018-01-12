import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

DATA_DIR = 'data/debug_data'
ATP_DIR = 'atp'
LOG_FILE = __file__.replace('.py', '.log')
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
train_theorems = set(proofs_train)
test_theorems = set(proofs_test)
params_rankings = {'chronology': chronology}
rankings_test = prs.Rankings(test_theorems, model=None, params=params_rankings,
                             n_jobs=N_JOBS, logfile=LOG_FILE)
proofs_test = prs.atp_evaluation(rankings_test, statements, dirpath=ATP_DIR,
                                 n_jobs=N_JOBS, logfile=LOG_FILE)
print(proofs_test.proofs)
