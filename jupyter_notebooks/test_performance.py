import os
import pickle

import torch

from dataset_embedding_cacher.embedding_cacher import EmbeddingCacher
from embedders.prottrans import ProtTransEmbedder
from model_training.dataset.cc_prediction_dataset import CCPredictionDataset
import tqdm
import matplotlib.pyplot as plt



first_feature = []

dataset = CCPredictionDataset(
        csv_file_path='../input/input_own.csv',
        id_sequence_label_idx=0,
        max_seq_len=4096,
        # label_transform=label_transform_begin

)
cache_path = 'test_performance.file'
if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
                X, Y = pickle.load(f)
else:
        embedder = ProtTransEmbedder()
        embedder = EmbeddingCacher(sequence_embedder=embedder, cache_path='../cache_prottrans.tmp.h5')


        n = 0
        features = [[] for _ in range(1024)]
        for seq, label in tqdm.tqdm(dataset):
                n += len(seq)


        X = torch.zeros(size=(n, 1024), dtype=torch.float)
        Y = torch.zeros(size=(n, 1), dtype=torch.float)


        n_running = 0
        for seq, label in tqdm.tqdm(dataset):

                embedding = embedder(seq)

                X[n_running:n_running+len(seq)] = embedding
                Y[n_running:n_running+len(seq), 0] = label

                n_running += len(seq)


        with open(cache_path, 'wb') as f:
                pickle.dump((X,Y), f)

plt.hist(X[:, 512     ].numpy(), 40)
plt.savefig('test.png')

from torch.distributions.multivariate_normal import MultivariateNormal

#cov = torch.cov(dat.transpose(0, 1))
#mn = MultivariateNormal(loc = dat.mean(dim=0), covariance_matrix=cov)

print('d')

from sklearn.ensemble import RandomForestClassifier

print("starting fitting")
rfc = RandomForestClassifier()
rfc.fit(X, Y.squeeze())

y_pred = rfc.predict(X)

import sklearn.metrics as metrics

print('ok')
print(metrics.f1_score(Y.squeeze(), y_pred))
print(rfc.score(X, Y.squeeze()))
print('done')
