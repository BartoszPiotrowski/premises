import sys, os, random
random.seed(1)
sys.path.append('..')
import premises as prs


N_JOBS = 45
DATA_DIR = 'data/MPTP2078'
OUTPUT_DIR = __file__.strip('.py')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
LOG_FILE = __file__.replace('.py', '.log')
ATP_DIR = os.path.join(OUTPUT_DIR, 'atp')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')


statements = prs.Statements(from_file=os.path.join(DATA_DIR, 'statements'),
                            logfile=LOG_FILE)
features = prs.Features(from_file=os.path.join(DATA_DIR, 'features'),
                            logfile=LOG_FILE)
chronology = prs.Chronology(from_file=os.path.join(DATA_DIR, 'chronology'),
                            logfile=LOG_FILE)
theorems = prs.utils.readlines(os.path.join(DATA_DIR, 'theorems_atpproved'))

params_data_trans = {'features': features,
                     'num_of_features': 0.5,
                     'chronology': chronology,
                     'ratio_neg_pos': 10,
                     'merge_mode': 'concat'}

# randomly generated rankings
rankings_random = prs.Rankings(theorems, model=None, params=params_data_trans,
                             n_jobs=N_JOBS, logfile=LOG_FILE)

params_atp_eval = {}
proofs = prs.atp_evaluation(rankings_random, statements, params=params_atp_eval,
                            dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)

params_train = {'model': 'network',
                'activation': 'relu',
                'batch_size': 100,
                'learning_rate': 0.01,
                'epochs': 50,
                'layers': 3,
                'hidden_layer': 100,
                'dropout': 0.3}
model_path = None
for i in range(30):
    prs.utils.printline("ITERATION: {}".format(i), LOG_FILE)
    train_labels, train_array = prs.proofs_to_train(proofs, params_data_trans,
                                           n_jobs=N_JOBS, logfile=LOG_FILE)

    params_train['num_of_features'] = train_array.shape[1]
    model_path = prs.train(train_labels, train_array, params=params_train,
           pretrained_model_path=model_path, model_dir=MODEL_DIR, n_jobs=N_JOBS,
                           logdir=LOG_DIR, logfile=LOG_FILE)

    pretrained_model_path = model_path
    rankings = prs.Rankings(theorems, model_path, params_data_trans,
                             model_type=params_train['model'],
                             n_jobs=N_JOBS, logfile=LOG_FILE)

    proofs.update(prs.atp_evaluation(rankings, statements, params_atp_eval,
                         dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE))

    proofs.print_stats(logfile=LOG_FILE)
    params_data_trans['rankings_for_negative_mining'] = rankings
    params_data_trans['level_of_negative_mining'] = 'all'

