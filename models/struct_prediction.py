import os.path
from typing import List, Tuple

import Bio
import numpy as np
import torch, esm
from samcc import SamCC
from typing import List
from Bio.PDB.Atom import Atom
from Bio.PDB.Model import Model
from Bio.PDB.DSSP import DSSP
import numpy as np
from torch import nn
import utils


class StructPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = esm.pretrained.esmfold_v1(
        )
        self.model.eval()

        self.dssp_path =  ''
        self.socket_path = ''
        self.sc = SamCC(bin_paths={'dssp':self.dssp_path, 'socket':self.socket_path})

        if not os.path.isdir('cache'):
            os.makedirs('cache')


    def forward(self, sequences: List[str]):
        if isinstance(sequences, str):
            sequences = [sequences]

        outputs = []

        for sequence in sequences:
            pdb_path = f"cache/{sequences}.pdb"
            out_path = f"cache/{sequences}.samcc"

            if not os.path.exists(pdb_path):
                output = self.model.infer_pdb(sequence)
                with open(pdb_path, "w") as f:
                    f.write(output)
                self.sc.run_samcc_turbo(pdbpath=pdb_path, outpath='/path/to/results',  save_pse=False)

                output = self.get_output(sequence=sequence, samcc_path = out_path)

                outputs.append(output)

        return outputs


    def get_output(self, sequence, samcc_path):
        out = torch.zeros(size=(sequence,), dtype=torch.float32)

        return out


    def get_data(self, pdb_path, id = None) -> Tuple[np.array, np.array, np.array]:
        return utils.get_data(pdb_path=pdb_path, dssp_path=self.dssp_path, id=id)


