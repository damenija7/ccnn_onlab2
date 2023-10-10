import csv
import os
import pickle
import re
import textwrap
from typing import Callable, List

import pandas
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import Subset

class CCPredictionDatasetPerResidue(Dataset):
    def __init__(
        self,
        csv_file_path: str,
        **kwargs
    ):
        # "uniprot_id", "residue_idx", "embedding", "label")
        #self.dat: pandas.DataFrame = pandas.read_csv(csv_file_path, chunksize=9000)


        self.csv_file_path = csv_file_path
        # Row start positions in file when reading rows
        self.row_file_pos = []
        self.labels = []
        self.siz = 0
        self.pos_rate = None


        self.chunksize = 9000

        if os.path.exists(self.csv_file_path + ".cache"):
            with open (self.csv_file_path + ".cache", "rb") as p_f:
                self.row_file_pos, self.pos_rate, self.labels = pickle.load(p_f)
                self.siz = len(self.row_file_pos)
        else:
            pos = 0
            neg = 0

            f = open(csv_file_path, "r")
            # skip header
            line = f.readline()
            while line:
                # skip empty lines
                line_pos = f.tell()
                line = f.readline()
                if len(line) < 5:
                    continue

                # Record start position for row in file
                self.row_file_pos.append(line_pos)
                self.siz += 1




                label = int(line[line.rindex(',')+1:])
                if label > 0:
                    pos += 1
                    self.labels.append(1)
                else:
                    neg += 1
                    self.labels.append(0)

            f.close()

            self.pos_rate = pos / (pos+neg)

            with open(self.csv_file_path + ".cache", "wb") as p_f:
                pickle.dump((self.row_file_pos, self.pos_rate, self.labels), p_f)






    def __len__(self):
        return self.siz

    def __getitem__(self, index):
        #row = self.dat.iloc[index]

        with open(self.csv_file_path) as f:
            f.seek(self.row_file_pos[index])
            # "uniprot_id", "residue_idx", "embedding", "label")
            line = f.readline().strip()
        #uniprot_id, residue_idx, embedding, label = re.search(r"(.*),(.*),\[(.*)\],(.*)", line).groups()
        label_idx = line.rindex(',')
        label = line[(label_idx+1):]

        embedding = line[line.index('['):line.rindex(']')+1]


        embedding = torch.Tensor([float(digit) for digit in embedding[1:-1].split(',')])
        label = torch.Tensor([float(label)])
        return embedding, label

    @staticmethod
    def _set_attributes(subset: Subset):
        P = sum(subset.dataset.labels[index] for index in subset.indices)
        N = len(subset.dataset) - P


        subset.pos_rate = P / (P + N)
        subset.labels = [subset.dataset.labels[i] for i in subset.indices]
