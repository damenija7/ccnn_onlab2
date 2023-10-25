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
from model_training.loss import DetrLoss


class TrainerRange:
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            weighted_random_sampler: bool = False
    ):
        self.train_loader, self.val_loader = train_loader, val_loader
        self.sequence_embedder: Callable = None

        # If True uses oversampling instead of class weighting to compensate for unbalanced dataset
        self.weighted_random_sampler = weighted_random_sampler


        self.transform_train_sequences = lambda sequences: [self.sequence_embedder.embed(sequence) for sequence in
                                                            sequences]

        self.transform_val_sequences = lambda sequences: [self.sequence_embedder.embed(sequence) for sequence in
                                                            sequences]



    def train(
            self, model: nn.Module, embedder: Callable, num_epochs: int, best_model_save_file_path: str
    ) -> Dict[str, Any]:
        train_stats, val_stats = {}, {}
        best_loss = math.inf

        self.sequence_embedder = embedder

        optimizer = optim.Adam(model.parameters())
        # optimizer = optim.RMSprop(model.parameters())

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
                f"\nTrain_Loss: {train_stats['loss'][-1]}, Val_Loss: {val_stats['loss'][-1]}, Val_Accuracy: {val_stats['accuracy'][-1]} Val_Sensitivity: {val_stats['sensitivity'][-1]}, Val_Precision: {val_stats['precision'][-1]}, Val_F1: {val_stats['F1'][-1]}"
            )

        return {
            "train": train_stats,
            "val": val_stats
        }

    def train_epoch(self, model, optimizer, tqdm_training_batch):
        train_epoch_stats = {'loss': [], 'accuracy': [], 'sensitivity': [], 'precision': [], 'F1': []}

        if self.previous_training_sensitivity is None:
            self.previous_training_sensitivity = 0.0
        # self.previous_training_sensitivity = 0.0

        device = next(model.parameters()).device

        model.train()
        for sequences, labels in tqdm_training_batch:
            # torch.cuda.empty_cache()
            # debug
            sequences_orig, labels_orig = sequences, labels

            self.current_batch_neg_rate = 1.0 - sum(label.mean() for label in labels) / len(labels)
            labels  = [label.to(device) for label in labels]

            seq_lens = [len(seq) for seq in sequences]
            self.set_current_batch_padding_mask(labels, seq_lens)

            sequences = self.transform_train_sequences(sequences)
            sequences = [sequence.to(device) for sequence in sequences]

            # reset gradients stored in optimizer
            optimizer.zero_grad()

            # get predictions of model
            preds = model(sequences, labels)

            if isinstance(preds, tuple):
                preds, loss = preds
            else:
                loss = self._get_loss(labels=labels, preds=preds, validation=False)

            if torch.isnan(preds).any():
                raise Exception("Model generated prediction contains nan")

            # get gradients
            try:
                loss.backward()
            except:
                pass

            # update weights
            optimizer.step()

            predicted_labels, targets_labels = self.get_non_range_labels(labels, preds, sequences)

            train_batch_stats = self.get_batch_stats(
                preds=predicted_labels, labels=targets_labels
            )


            train_batch_stats['loss'] = loss.item()
            try:
                self.previous_training_sensitivity = self.previous_training_sensitivity * (1 - 0.0025) + float(
                    train_batch_stats['sensitivity']) * 0.0025
            except:
                pass

            for key, val in train_batch_stats.items():
                if val > -1:
                    train_epoch_stats.setdefault(key, []).append(val)

            tqdm_training_batch.set_description(
                f'Training Batch ({OrderedDict((metric_name, "{:1.4f}".format(metric_val) if metric_val >= 0 else "NONE") for metric_name, metric_val in train_batch_stats.items())}')

        return {key: np.mean(train_epoch_stats[key]) if len(train_epoch_stats[key]) > 0 else 0.0 for key in
                train_epoch_stats}

    def validate_epoch(self, model):
        val_epoch_stats = {'loss': [], 'accuracy': [], 'sensitivity': [], 'precision': [], 'F1': []}

        if self.previous_validation_sensitivity is None:
            self.previous_validation_sensitivity = 0.0
        # self.previous_validation_sensitivity = 0.0

        device = next(model.parameters()).device

        model.eval()
        tqdm_val_batch = tqdm(self.val_loader, "Validation Batch")
        for sequences, labels in tqdm_val_batch:
            # torch.cuda.empty_cache()

            self.current_batch_neg_rate = 1.0 - sum(label.mean() for label in labels) / len(labels)
            labels = [label.to(device) for label in labels]

            seq_lens = [len(seq) for seq in sequences]
            self.set_current_batch_padding_mask(labels, seq_lens)

            sequences = self.transform_val_sequences(sequences)
            sequences = [sequence.to(device) for sequence in sequences]

            with torch.no_grad():
                preds = model(sequences)

                if isinstance(preds, tuple):
                    preds, loss = preds
                    loss = loss.item()
                else:
                    loss = self._get_loss(labels=labels, preds=preds, validation=False).item()

                predicted_labels, targets_labels = self.get_non_range_labels(labels, preds, sequences)
                val_batch_stats = self.get_batch_stats(
                    preds=predicted_labels, labels=targets_labels
                )
                val_batch_stats['loss'] = loss
                try:
                    self.previous_validation_sensitivity = self.previous_validation_sensitivity * (1 - 0.001) + float(
                        val_batch_stats['sensitivity']) * 0.001 if not type(val_batch_stats['sensitivity'],
                                                                            str) else self.previous_validation_sensitivity
                except:
                    pass

                for key, val in val_batch_stats.items():
                    if val > -1:
                        val_epoch_stats.setdefault(key, []).append(val)

                tqdm_val_batch.set_description(
                    f'Validation Batch ({OrderedDict((metric_name, "{:1.4f}".format(metric_val) if metric_val > 0 else "NONE") for metric_name, metric_val in val_batch_stats.items())}')

        return {key: np.mean(val_epoch_stats[key]) if len(val_epoch_stats[key]) > 0 else 0.0 for key in val_epoch_stats}

    def get_non_range_labels(self, labels, preds, sequences):
        with torch.no_grad():
            predicted_labels = [torch.zeros_like(seq[:, 0], dtype=torch.bool) for seq in sequences]
            targets_labels = [x.clone() for x in predicted_labels]
            for batch_idx, seq in enumerate(sequences):
                seq_len = len(seq)

                for center, width, probability in preds[batch_idx]:

                    if probability > 0.5:
                        left, right = self.get_cc_left_right_indices(center, seq_len, width)
                        predicted_labels[batch_idx][left:right] = True

                for center, width in labels[batch_idx]:
                    left, right = self.get_cc_left_right_indices(center, seq_len, width)

                    targets_labels[batch_idx][left:right] = True
        return predicted_labels, targets_labels

    def get_cc_left_right_indices(self, center, seq_len, width):
        center = seq_len * center.item()
        width = seq_len * width.item()

        left = max(int(math.ceil(center - width/2)), 0)
        right = min(int(math.ceil(center + width / 2)), seq_len)

        return left, right

    def set_current_batch_padding_mask(self, labels, seq_lens):
        self.label_padding_mask = torch.empty(
            size=(len(labels), max(len(label) for label in labels)), dtype=torch.bool, device=labels[0].device)

        for i, label in enumerate(labels):
            self.label_padding_mask[i, :len(label)] = True





    def _get_loss(self, labels: List[torch.Tensor], preds: torch.Tensor, validation: bool):
        dataset = self.train_loader.dataset if not validation else self.val_loader.dataset

        return DetrLoss(pos_rate=dataset.pos_rate)(preds=preds, targets=labels)

    def get_batch_stats(self, preds, labels) -> Dict[str, float]:
        with torch.no_grad():
            preds, labels = torch.cat(preds), torch.cat(labels)

            correct_preds = preds == labels

            i = (preds & labels).sum().item()
            u = (preds | labels).sum().item()
            iou = i/u if u > 0.0 else -1

            correct_preds_positive = correct_preds[labels > 0]
            correct_preds_negative = correct_preds[labels < 1]
            TP = correct_preds_positive.sum().item()
            FN = torch.numel(correct_preds_positive) - TP
            TN = correct_preds_negative.sum().item()
            FP = torch.numel(correct_preds_negative) - TN


        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else -1
        precision = TP / (TP + FP) if (TP + FP) > 0 else -1
        F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else -1

        # accuracy = (preds_round == labels).float().mean().item()
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'precision': precision,
            'F1': F1,
            'iou': iou
        }
