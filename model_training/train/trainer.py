import math
from collections import OrderedDict
from typing import Dict, Any, Callable, List

import numpy as np
import torch
import torchvision
from torch import nn, optim, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model_training.dataset.cc_prediction_dataset import CCPredictionDataset
from model_training.dataset.cc_prediction_dataset_per_residue import CCPredictionDatasetPerResidue
from models import rnn


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        sequence_embedder: Callable = lambda x: x,
        weighted_random_sampler: bool = False
    ):
        self.train_loader, self.val_loader = train_loader, val_loader
        self.sequence_embedder: Callable = sequence_embedder

        # If True uses oversampling instead of class weighting to compensate for unbalanced dataset
        self.weighted_random_sampler = weighted_random_sampler

        # Transform Sequence based on whether given dataset is sequence_residue/row or sequence/row
        #
        #
        if self.get_dataset_type(self.val_loader.dataset) == CCPredictionDataset:
            self.transform_val_sequences = lambda sequences : [self.sequence_embedder.embed(sequence) for sequence in sequences]
        else:
            self.transform_val_sequences = lambda sequences: [torch.unsqueeze(sequence, 0) for sequence in sequences]

        if self.get_dataset_type(self.train_loader.dataset) == CCPredictionDataset:
            self.transform_train_sequences = lambda sequences : [self.sequence_embedder(sequence) for sequence in sequences]
        else:
            self.transform_train_sequences = lambda sequences: [torch.unsqueeze(sequence, 0) for sequence in sequences]


        self.half = False

    def get_dataset_type(self, dataset):
        if isinstance(dataset, CCPredictionDataset):
            return CCPredictionDataset

        if not hasattr(dataset, 'dataset'):
            return CCPredictionDatasetPerResidue

        if isinstance(dataset.dataset, CCPredictionDataset):
            return CCPredictionDataset

        if not hasattr(dataset.dataset, 'dataset'):
            return CCPredictionDatasetPerResidue

        if isinstance(dataset.dataset.dataset, CCPredictionDataset):
            return CCPredictionDataset

        return CCPredictionDatasetPerResidue

    def train(
        self, model: nn.Module, embedder: Callable, num_epochs: int, best_model_save_file_path: str
    ) -> Dict[str, Any]:
        train_stats, val_stats = {}, {}
        best_loss = math.inf

        self.sequence_embedder = embedder

        optimizer = optim.Adam(model.parameters())
        #optimizer = optim.RMSprop(model.parameters())

        if self.half:
            model.half()
            optimizer = optim.Adam(model.parameters(), eps=10e-4)

        self.previous_training_sensitivity, self.previous_validation_sensitivity = None, None


        for epoch_idx in range(num_epochs):
            tqdm.write(f"At Epoch {epoch_idx}")

            tqdm_training_batch = tqdm(self.train_loader, "Training Batch")


            train_batch_stats = self.train_epoch(
                model, optimizer, tqdm_training_batch
            )

            # Keep track of Mean Training Stat Value of the current Epoch
            for key, val in train_batch_stats.items():
                if math.isnan(val) or math.isinf(val):
                    val = 0.0

                train_stats.setdefault(key, []).append(val)

            #
            # Validation of Current Epoch
            #
            val_batch_stats = self.validate_epoch(model)

            # save model if validation sensitivity is such that it is considered the best predictor
            if val_batch_stats['loss'] < best_loss:
                best_loss = val_batch_stats['loss']
                torch.save(model, best_model_save_file_path)

            # Record the Main Validation Loss and Validation Accuracy of the current Epoch
            for key, val in val_batch_stats.items():
                if math.isnan(val) or math.isinf(val):
                    val = 0.0

                val_stats.setdefault(key, []).append(val)

            # print acc, val
            tqdm.write(
                f"\nTrain_Loss: {train_stats['loss'][-1]}, Val_Loss: {val_stats['loss'][-1]}, Val_Accuracy: {val_stats['accuracy'][-1]} Val_Sensitivity: {val_stats['sensitivity'][-1]}, Val_Precision: {val_stats['precision'][-1]}, Val_F1: {val_stats['F1'][-1]}, Val_IOU: {val_stats['iou'][-1]}"
            )



        return {
            "train": train_stats,
            "val": val_stats
        }

    def train_epoch(self, model, optimizer, tqdm_training_batch):
        train_epoch_stats = {'loss': [], 'accuracy': [], 'sensitivity': [], 'precision': [], 'F1': []}

        if self.previous_training_sensitivity is None:
            self.previous_training_sensitivity = 0.0
        #self.previous_training_sensitivity = 0.0

        device = next(model.parameters()).device

        model.train()
        for sequences, labels in tqdm_training_batch:
            if all(not torch.any(label > 0) for label in labels):
                continue

            torch.cuda.empty_cache()
            # debug
            sequences_orig, labels_orig = sequences, labels

            self.current_batch_neg_rate = 1.0 - sum(label.mean() for label in labels) / len(labels)
            labels, labels_orig, sequences, sequences_orig = self.convert_seq_label(sequences, labels, device)

            # reset gradients stored in optimizer
            optimizer.zero_grad()

            # get predictions of model
            if isinstance(model, rnn.Transformer):
                preds = model(sequences_orig, labels_orig)
            else:
                preds = model(sequences, labels)

            if isinstance(preds, tuple):
                preds, loss = preds
            else:
                loss = self._get_loss(labels=labels, preds=preds, validation=False)


            if torch.isnan(preds).any():
                raise Exception("Model generated prediction contains nan")



            # get gradients
            loss.backward()

            # update weights
            optimizer.step()


            train_batch_stats = self.get_batch_stats(
                labels=labels, preds=preds, loss=loss
            )


            for key, val in train_batch_stats.items():
                if val > -1:
                    train_epoch_stats.setdefault(key, []).append(val)



            tqdm_training_batch.set_description(f'Training Batch ({OrderedDict((metric_name, "{:1.4f}".format(metric_val) if metric_val != -1 else "NONE") for metric_name, metric_val in train_batch_stats.items())}')

        return {key: np.mean(train_epoch_stats[key]) if len(train_epoch_stats[key]) > 0 else 0.0 for key in train_epoch_stats}

    def convert_seq_label(self, sequences, labels, device):
        labels_orig = [label.to(device) for label in labels]
        labels = pad_sequence(labels, batch_first=True).to(device)
        seq_lens = [len(seq) for seq in sequences]
        self.set_current_batch_padding_mask(labels, seq_lens)
        sequences = self.transform_train_sequences(sequences)
        if isinstance(sequences[0], str):
            return labels, labels_orig, sequences, sequences

        sequences_orig = [sequence.to(device) for sequence in sequences]
        sequences = pad_sequence(sequences, batch_first=True).to(device)

        if self.half:
            sequences = sequences.half()
            labels = labels.half()
            labels_orig = [label.half() for label in labels_orig]
            sequences_orig = [sequence.half() for sequence in sequences_orig]

        return labels, labels_orig, sequences, sequences_orig

    def set_current_batch_padding_mask(self, labels, seq_lens):
        self.current_batch_padding_mask = torch.empty(labels.shape[:2], dtype=torch.bool, device=labels.device)
        for i, seq_len in enumerate(seq_lens):
            self.current_batch_padding_mask[i, :seq_len] = True

    def validate_epoch(self, model):
        val_epoch_stats = {'loss': [], 'accuracy': [], 'sensitivity': [], 'precision': [], 'F1': []}


        device = next(model.parameters()).device


        model.eval()
        tqdm_val_batch = tqdm(self.val_loader, "Validation Batch")
        for sequences, labels in tqdm_val_batch:
            torch.cuda.empty_cache()

            self.current_batch_neg_rate = 1.0 - sum(label.mean() for label in labels) / len(labels)
            labels, labels_orig, sequences, sequences_orig = self.convert_seq_label(sequences, labels, device)

            with torch.no_grad():
                # get predictions of model
                preds = model(sequences, labels)

                if isinstance(preds, tuple):
                    preds, loss = preds
                else:
                    loss = self._get_loss(labels=labels, preds=preds, validation=True)

                if torch.isnan(preds).any():
                    raise Exception("Model generated prediction contains nan")

                val_batch_stats = self.get_batch_stats(
                    labels=labels, preds=preds, loss=loss
                )

                for key, val in val_batch_stats.items():
                    if val > -1:
                        val_epoch_stats.setdefault(key, []).append(val)


                tqdm_val_batch.set_description(f'Validation Batch ({OrderedDict((metric_name, "{:1.4f}".format(metric_val) if metric_val >= 0 else "NONE") for metric_name, metric_val in val_batch_stats.items())}')

        return {key: np.mean(val_epoch_stats[key]) if len(val_epoch_stats[key]) > 0 else 0.0 for key in val_epoch_stats}


    def _get_loss(self, labels: torch.Tensor, preds: torch.Tensor, validation: bool):
        # Weight positive label error with Negative Label Rate so that Positive Prediction becomes much more important due to its scarcity in training ! (unbalanced datset)
        #loss = torchvision.ops.focal_loss.sigmoid_focal_loss(inputs=preds, targets=labels, alpha=1.0 - self.train_loader.dataset.pos_rate, reduction="sum")

        # Only consider preds for non-padded regions in batch ( i.e. preds for actual sequences, not paddings)
        preds = preds[self.current_batch_padding_mask]
        labels = labels[self.current_batch_padding_mask]

        #previous_sensitivity = self.previous_validation_sensitivity if validation else self.previous_training_sensitivity
        #loss = torchvision.ops.focal_loss.sigmoid_focal_loss(inputs=preds, targets=labels,
        #                                                    alpha=previous_sensitivity * 0.5 + (1-previous_sensitivity) * 0.999 ,
        #reduction="mean")

        #if not hasattr(self, 'losser'):
        #    self.losser = torch.nn.BCELoss(reduction='mean')
        #loss = torch.nn.BCELoss(reduction='mean')(preds, labels)

        #loss = torchvision.ops.focal_loss.sigmoid_focal_loss(inputs=preds, targets=labels,
        #                                                     alpha=1.0 - (self.val_loader.dataset.pos_rate if validation else self.train_loader.dataset.pos_rate),
        #                                              reduction="mean")


        labels = torch.squeeze(labels, dim=-1)
        preds = torch.squeeze(preds, dim=-1)

        # Compensate for unbalanced dataset property by class weighting
        if not self.weighted_random_sampler:
            if validation:
                # Using class weighting in the loss fn during validation leads to skewed (positive-favored) values
                # That can lead to incorrect interpretations at the user's end
                #return (nn.BCELoss()(preds, labels))
                return (nn.BCELoss()(preds, labels))
            else:
                pos_rate = self.train_loader.dataset.pos_rate
                weights = torch.full_like(preds, pos_rate)
                weights[labels > 0] = 1 - pos_rate
                return nn.BCELoss(weight=weights)(preds, labels)


        # When compensate for unbalanced dataset property by oversampling (instead of loss function)
        return nn.BCELoss(reduction='mean')(preds, labels)

    def get_batch_stats(self, preds, labels, loss) -> Dict[str, float]:
        with torch.no_grad():
            labels, preds = labels.squeeze(dim=-1), preds.squeeze(dim=-1)

            if preds.shape[0] != labels.shape[0] or preds.shape[1] != labels.shape[1]:
                return {
                    'accuracy': -1,
                    'sensitivity': -1,
                    'precision': -1,
                    'F1': -1,
                    'iou': -1,
                    'loss': loss.item()
                }

            preds_round = preds.round()


            correct_preds = preds_round == labels

            i = ((preds_round * labels) > 0).sum().item()
            u = ((preds_round + labels) > 0).sum().item()
            iou = i / u if u > 0.0 else -1


            correct_preds_positive = correct_preds[labels > 0]
            correct_preds_negative = correct_preds[labels < 1]
            TP = correct_preds_positive.sum().item()
            FN = torch.numel(correct_preds_positive) - TP
            TN = correct_preds_negative.sum().item()
            FP = torch.numel(correct_preds_negative) - TN

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else -1
        precision = TP / (TP + FP) if (TP + FP) > 0 else -1
        F1 = 2 * TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else -1

        # accuracy = (preds_round == labels).float().mean().item()
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'precision': precision,
            'F1': F1,
            'iou': iou,
            'loss': loss.item()
        }
