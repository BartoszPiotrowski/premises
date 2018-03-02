import sys
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 10
DATA_DIR = 'data/MPTP2078'
ATP_DIR = 'atp'
RANKINGS_DIR = 'rankings'
LOG_FILE = __file__.replace('.py', '.log')

rankings = prs.utils.load_obj(join(RANKINGS_DIR, rankings_multi_mizar40_1.pkl))
for r in rankings:
    print(r)
    prs.utils.writelines(rankings[r], r)
