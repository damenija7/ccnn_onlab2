import os
import pickle
from pprint import pprint

import torch

from dataset_embedding_cacher.embedding_cacher import EmbeddingCacher
from embedders.prottrans import ProtTransEmbedder
from model_training.dataset.cc_prediction_dataset import CCPredictionDataset
from model_training.dataset.collator import raw_collator
from model_training.train.trainer import Trainer

#input
saved_model_path: str = '../output/training_results_2023_10_27_17_05_20'
validation_file: str = '../input/deepcoil_test1.csv'
cache_path: str = '../cache_prottrans.tmp.h5'

model_path = saved_model_path + '/' + [model_path for model_path in os.listdir(saved_model_path) if model_path.endswith('.pkl')][0]


model = torch.load(model_path)

print("loading embedder..")
embedder = EmbeddingCacher(sequence_embedder=ProtTransEmbedder(), cache_path=cache_path)

print("loading dataset..")
dataset = CCPredictionDataset(
        csv_file_path=validation_file,
        id_sequence_label_idx=0,
        max_seq_len=2048,
        # label_transform=label_transform_begin
        seq_embedder=embedder
)

from torch.utils.data import DataLoader
val_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=raw_collator, sampler=None, drop_last=True
    )


trainer = Trainer(
    train_loader = val_loader,
    val_loader = val_loader,
    sequence_embedder=embedder
)

model.eval()
model.to(torch.device('cuda'))
stats = trainer.validate_epoch(model)
pprint(stats)

