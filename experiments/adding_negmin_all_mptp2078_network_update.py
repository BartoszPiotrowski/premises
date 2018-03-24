import sys, os
sys.path.append('..')
import premises as prs

N_JOBS = 20
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
proofs_train = prs.Proofs(from_file=os.path.join(DATA_DIR, 'atpproved.train'),
                          logfile=LOG_FILE)
theorems = prs.utils.readlines(os.path.join(DATA_DIR, 'theorems_atpproved'))
train_theorems = set(proofs_train)
test_theorems = set(theorems) - set(train_theorems)

params_data_trans = {'features': features,
                     'num_of_features': 0.2,
                     'chronology': chronology,
                     'ratio_neg_pos': 10,
                     'merge_mode': 'concat'}

train_labels, train_array = prs.proofs_to_train(proofs_train, params_data_trans,
                                       n_jobs=N_JOBS, logfile=LOG_FILE)
params_train = {'model': 'network',
                'activation': 'relu',
                'batch_size': 100,
                'learning_rate': 0.001,
                'epochs': 100,
                'layers': 1,
                'hidden_layer': 100,
                'dropout': 0.3}
params_train['num_of_features'] = train_array.shape[1]
model = prs.train(train_labels, train_array, params=params_train,
                           model_dir=MODEL_DIR, n_jobs=N_JOBS,
                           logdir=LOG_DIR, logfile=LOG_FILE)
rankings_train = prs.Rankings(train_theorems, model, params_data_trans,
                             model_type=params_train['model'],
                     n_jobs=N_JOBS, logfile=LOG_FILE)
rankings_test = prs.Rankings(test_theorems, model, params_data_trans,
                             model_type=params_train['model'],
                     n_jobs=N_JOBS, logfile=LOG_FILE)
params_atp_eval = {}
proofs_train.update(prs.atp_evaluation(rankings_train, statements,
           params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE))
prs.utils.printline("STATS OF TRAINING PROOFS", logfile=LOG_FILE)
proofs_train.print_stats(logfile=LOG_FILE)
proofs_test = prs.atp_evaluation(rankings_test, statements, params_atp_eval,
                                 dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE)
prs.utils.printline("STATS OF TEST PROOFS", logfile=LOG_FILE)
proofs_test.print_stats(logfile=LOG_FILE)
params_data_trans['level_of_negative_mining'] = 'all'
model = None
for i in range(40):
    pretrained_model_path = model
    prs.utils.printline("NEGATIVE MINING ROUND: {}".format(i + 1),
                        logfile=LOG_FILE)
    params_data_trans['rankings_for_negative_mining'] = rankings_train
    train_labels, train_array = prs.proofs_to_train(proofs_train,
                    params_data_trans, n_jobs=N_JOBS, logfile=LOG_FILE)
    model = prs.train(train_labels, train_array, params=params_train,
                            pretrained_model_path=model,
                           model_dir=MODEL_DIR, n_jobs=N_JOBS,
                           logdir=LOG_DIR, logfile=LOG_FILE)
    rankings_train = prs.Rankings(train_theorems, model, params_data_trans,
                             model_type=params_train['model'],
                         n_jobs=N_JOBS, logfile=LOG_FILE)
    rankings_test = prs.Rankings(test_theorems, model, params_data_trans,
                             model_type=params_train['model'],
                         n_jobs=N_JOBS, logfile=LOG_FILE)
    proofs_train.update(prs.atp_evaluation(rankings_train, statements,
          params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE))
    prs.utils.printline("STATS OF TRAINING PROOFS", logfile=LOG_FILE)
    proofs_train.print_stats(logfile=LOG_FILE)
    proofs_test.update(prs.atp_evaluation(rankings_test, statements,
         params_atp_eval, dirpath=ATP_DIR, n_jobs=N_JOBS, logfile=LOG_FILE))
    prs.utils.printline("STATS OF TEST PROOFS", logfile=LOG_FILE)
    proofs_test.print_stats(logfile=LOG_FILE)