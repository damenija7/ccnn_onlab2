import csv
import textwrap
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CCPredictionDatasetRange(Dataset):
    def __init__(
        self,
        csv_file_path: str,
        id_transform: Callable = lambda x: x,
        sequence_transform: Callable = lambda x: x,
        delimiter: str = ",",
        skip_header: bool = True,
        id_sequence_label_idx: int = 0,
        max_seq_len: int = 512,
    ):
        self.prot_id_list, self.prot_sequence_list, self.prot_labels_list = [], [], []

        # Will have to calculate
        self.pos_rate = 0.0
        num_pos_residue_label = 0
        num_neg_residue_label = 0

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

                row_pos = prot_label.count('1')
                row_neg = len(prot_label) - row_pos
                num_pos_residue_label += row_pos
                num_neg_residue_label += row_neg

                self.prot_id_list.append(id_transform(prot_id)),
                self.prot_sequence_list.append(sequence_transform(prot_sequence))

                # left closed, right open [, ) indexing
                prot_label_ranges = []
                # self.prot_labels_list.append(label_transform(prot_label))
                # handle label langes
                range_start = None
                for residue_idx, residue_label in enumerate(prot_label):
                    if residue_label == '1':
                        if not range_start:
                            range_start = residue_idx
                    elif residue_label == '0':
                        if range_start:
                            prot_label_ranges.append((range_start, residue_idx))
                            range_start = None
                # If cc region at end of sequence
                if range_start:
                    prot_label_ranges.append((range_start, len(prot_label)))

                self.prot_labels_list.append(prot_label_ranges)

        self.pos_rate = num_pos_residue_label / (num_pos_residue_label + num_neg_residue_label)

    def __len__(self):
        return len(self.prot_labels_list)

    def __getitem__(self, index):
        return self.prot_sequence_list[index], self.prot_labels_list[index]
