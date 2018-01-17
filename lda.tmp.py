import gensim
import numpy as np

m = np.random.randint(2, size=(20, 10))
a=gensim.matutils.Dense2Corpus(m,documents_columns=False)
lda = models.LdaModel(a, id2word = None, num_topics = 3)
gensim.matutils.corpus2dense(lda[a], 3).T

