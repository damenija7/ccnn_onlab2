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
                dat = pickle.load(f)
else:
        embedder = ProtTransEmbedder()
        embedder = EmbeddingCacher(sequence_embedder=embedder, cache_path='../cache_prottrans.tmp.h5')


        n = 0
        features = [[] for _ in range(1024)]
        for seq, label in tqdm.tqdm(dataset):
                n += len(seq)


        dat = torch.zeros(size=(n, 1024), dtype=torch.float)


        n_running = 0
        for seq, label in tqdm.tqdm(dataset):

                embedding = embedder(seq)

                dat[n_running:n_running+len(seq)] = embedding

                n_running += len(seq)


        with open(cache_path, 'wb') as f:
                pickle.dump(dat, f)

plt.hist(dat[:, 0].numpy(), 40)
plt.show()
plt.savefig('test.png')