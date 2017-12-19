from os import path.join
import sys
import premises as prs

from random import sample

DATA_DIR = 'data/debug_data'

features = prs.Features(from_file=path.join(DATA_DIR, 'features'))
#print(type(features))
#print(len(features))
#print(features['t104_tmap_1'])

proofs_train = prs.Proofs(from_file=path.join(DATA_DIR, 'atpproved.train'))
#print(len(proofs_train))
#print(proofs_train['t104_tmap_1'])
#print('t104_tmap_1' in proofs_train)

chronology = prs.Chronology(from_file=path.join(DATA_DIR, 'chronology'))
#print(len(chronology))
#print(chronology.index('t104_tmap_1'))
#print('t104_tmap_1' in chronology)

statements = prs.Statements(from_file=path.join(DATA_DIR, 'statements'))
#print(len(statements))
#print(statements['t104_tmap_1'])

test_theorems = set(sample(set(chronology) - set(proofs_train), 3))
#print(len(test_theorems))

params_data_trans = {'features': features,
                     'features_ordered': features.all_features(),
                     'chronology': chronology,
                     'sparse': False}

train_labels, train_array = prs.proofs_to_train(proofs_train, params_data_trans)
#print(train_array.shape)
#print(len(train_labels))

params_train = {}
model = prs.train(train_labels, train_array, params=params_train)

rankings_train = prs.Rankings(test_theorems, model, params_data_trans)
#print(len(rankings_train))

params_atp_eval = {}
# TODO atp eval without rankings
proofs_test = prs.atp_evaluation(rankings_train, statements, params_atp_eval)
print(proofs_test)

sys.exit()
