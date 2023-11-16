import os.path
import warnings
from hashlib import sha256
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
from utils.struct_prediction import get_data_struct
from utils.struct_prediction_socket import get_socket_data
from utils.struct_prediction_twister import get_twister_data


class StructPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = esm.pretrained.esmfold_v1(
        )
        self.model.eval()

        self.device = self.model.device

        self.dssp_path =  'utils/dssp-x86_64.AppImage'
        self.socket_path = ''
        self.sc = SamCC(bin_paths={'dssp':self.dssp_path, 'socket':self.socket_path})

        if not os.path.isdir('cache'):
            os.makedirs('cache')

    def to(self, device):
        super().to(device)
        self.device = device
        self.model = self.model.to(device)


    def forward(self, sequences: List[str], labels = None):
        if isinstance(sequences, str):
            sequences = [sequences]

        outputs = []

        for sequence in sequences:
            hashed = sha256(sequence.encode()).hexdigest()
            pdb_path = f"cache/{hashed}.pdb"
            out_path = f"cache/{hashed}.samcc"

            if not os.path.exists(pdb_path):
                pdb_oudput = self.model.infer_pdb(sequence)
                with open(pdb_path, "w") as f:
                    f.write(pdb_oudput)


            output = self.get_output(pdb_path = pdb_path, samcc_path = out_path)
            outputs.append(output)

        res = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True).to(self.device).type(torch.float)
        return res, torch.tensor([0.0], requires_grad=True)


    def get_output(self, pdb_path, samcc_path):
        warnings.filterwarnings("error")
        data_struct = get_data_struct(pdb_path, self.dssp_path)

        data_socket = get_socket_data(data_struct)
        data_twister = get_twister_data(data_struct, data_socket)

        cc_mask = data_socket['cc_mask_by_model'][0]
        cc_mask |= data_twister['cc_mask_by_model'][0]

        warnings.resetwarnings()

        return data_twister


    def get_data(self, pdb_path, id = None) -> Tuple[np.array, np.array, np.array]:
        return utils.get_data(pdb_path=pdb_path, dssp_path=self.dssp_path, id=id)


