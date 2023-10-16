import csv
import sys
import textwrap
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CCPredictionDatasetForceResidue(Dataset):
    def __init__(
        self,
        csv_file_path: str,
        id_transform: Callable = lambda x: x,
        sequence_transform: Callable = lambda x: x,
        label_transform: Callable = lambda label: torch.Tensor(
            [float(digit) for digit in label]
        ),
        delimiter: str = ",",
        skip_header: bool = True,
        id_sequence_label_idx: int = 0,
        seq_embedder = Callable,
            **kwargs
    ):
        self.seq_embedder = seq_embedder

        self.sequences, self.labels, self.ids = [], [], []

        self.num_total_residues = 0

        self.idx_to_id_and_res_idx = []

        with open(csv_file_path, newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)

            if skip_header:
                next(csv_reader, None)

            for row_idx, row in enumerate(csv_reader):
                prot_id, prot_sequence, prot_label = (
                    row[id_sequence_label_idx + 0],
                    row[id_sequence_label_idx + 1],
                    row[id_sequence_label_idx + 2]
                )

                self.ids.append(id_transform(prot_id))
                self.sequences.append(sequence_transform(prot_sequence))
                self.labels.append(label_transform(prot_label))

                for residue_idx, residue in enumerate(prot_sequence):
                    self.num_total_residues += 1
                    self.idx_to_id_and_res_idx.append((row_idx, residue_idx))




        self._set_attributes(self)



    def __len__(self):
        return self.num_total_residues

    def __getitem__(self, index):
        row_idx, residue_idx = self.idx_to_id_and_res_idx[index]
        seq, label = self.sequences[row_idx], self.labels[row_idx]

        return self.seq_embedder([seq])[0][residue_idx], torch.unsqueeze(label[residue_idx], dim=-1)

    @staticmethod
    def _set_attributes(dataset: Dataset):
        P, N = 0, 0

        pos = []

        for seq, label in zip(dataset.sequences, dataset.labels):
            # #if len(label.shape) < 2:
            # #    label = torch.unsqueeze(label, dim=-1)
            #
            # seq_P = 0
            # # TDOO TMP
            # for res_idx in range(label.shape[0]):
            #     if torch.all(label[res_idx] == 0.0):
            #         N += 1
            #     else:
            #         seq_P += 1
            #         P += 1
            # pos.append(seq_P)

            seq_P = torch.sum(label).item()
            seq_N = torch.numel(label) - seq_P


            for residue in label:
                pos.append(residue)

            P += seq_P
            N += seq_N


        dataset.pos = pos
        dataset.pos_rate = P / (P + N)
        # dataset.labels = [label for _, label in dataset]
