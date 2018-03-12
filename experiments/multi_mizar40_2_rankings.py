import sys
import xgboost as xgb
from os.path import join
from random import sample
sys.path.append('..')
import premises as prs

N_JOBS = 52
DATA_DIR = 'data/mizar40'
MODELS_DIR = 'models'
RANKINGS_DIR = 'rankings'
RANKINGS_FILES_DIR = 'rankings_2'
LOG_FILE = __file__.replace('.py', '.log')

features = prs.Features(from_file=join(DATA_DIR, 'features'), logfile=LOG_FILE)
chronology = prs.Chronology(from_file=join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
test_theorems = prs.utils.readlines(join(DATA_DIR, 'theorems_not_atpproved'))
params_data_trans = {'features': features,
                     'num_of_features': int(0.3 * features.num_of_features),
                     'merge_mode': 'concat',
                     'chronology': chronology}
# model.save_model(join(MODELS_DIR, 'model_multi_mizar40_2.pkl'))
model = xgb.Booster()
model.load_model(join(MODELS_DIR, 'model_multi_mizar40_2.pkl'))
rankings_test = prs.Rankings(test_theorems, model, params_data_trans,
             save_to_dir=RANKINGS_FILES_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
prs.utils.save_obj(rankings_test,
                   join(RANKINGS_DIR, 'rankings_multi_mizar40_2.pkl'))
