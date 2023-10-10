import csv
import sys
import textwrap
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CCPredictionDataset(Dataset):
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
        max_seq_len: int = sys.maxsize,
    ):
        self.ids, self.sequences, self.labels = [], [], []
        self.pos = []

        with open(csv_file_path, newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)

            if skip_header:
                next(csv_reader, None)

            for row in csv_reader:
                prot_id, prot_sequence, prot_label = (
                    row[id_sequence_label_idx + 0],
                    row[id_sequence_label_idx + 1],
                    row[id_sequence_label_idx + 2],
                )

                if len(prot_sequence) > max_seq_len:
                    continue


                self.ids.append(id_transform(prot_id)),
                self.sequences.append(sequence_transform(prot_sequence))
                self.labels.append(label_transform(prot_label))

        self._set_attributes(self)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]

    @staticmethod
    def _set_attributes(dataset: Dataset):
        P, N = 0, 0

        pos = []

        for seq, label in dataset:
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
            pos.append(seq_P)

            P += seq_P
            N += seq_N


        dataset.pos = pos
        dataset.pos_rate = P / (P + N)
        dataset.labels = [label for _, label in dataset]
