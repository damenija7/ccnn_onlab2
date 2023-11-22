import os.path
import warnings
from hashlib import sha256
from typing import List, Tuple, Optional

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
        self.model = esm.pretrained.esmfold_v0(
        )
        self.model.eval()

        self.device = self.model.device

        self.dssp_path =  '/home/damenija7/.local/bin/mkdssp'
        self.socket_path = '/home/damenija7/.local/bin/socket2'
        self.pcasso_path = 'utils/pcasso'
        self.sc = SamCC(bin_paths={'dssp':self.dssp_path, 'socket':self.socket_path})


        self.dummy_header = 'HEADER    HYDROLASE                               26-SEP-13   3WIW              '

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

        for seq_idx, sequence in enumerate(sequences):
            hashed = sha256(sequence.encode()).hexdigest()
            pdb_path = f"cache/{hashed}.pdb"
            out_path = f"cache/{hashed}.samcc"

            if not os.path.exists(pdb_path):
                pdb_output = self.model.infer_pdb(sequence)
                with open(pdb_path, "w") as f:

                    if not pdb_output.startswith("HEADER"):
                        f.write(self.dummy_header + '\n' + pdb_output)
                    else:
                        f.write(pdb_output)


            output = self.get_output(pdb_path = pdb_path, samcc_path = out_path, label=labels[seq_idx] if labels is not None else None)
            outputs.append(output)

        res = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True).to(self.device).type(torch.float)
        return res, torch.tensor([0.0], requires_grad=True)


    def get_output(self, pdb_path, samcc_path, label: Optional[torch.Tensor] = None):
        # warnings.filterwarnings("error")
        warnings.simplefilter('ignore')
        data_struct = get_data_struct(pdb_path, self.dssp_path, self.pcasso_path)
        num_res: int = data_struct['alpha_carbon_coords_by_model'][0].shape[0]

        if label is not None:
            alpha_helix_mask = torch.zeros(size=(num_res,), dtype=torch.bool).to(label.device)
            for start, end in data_struct['alpha_helix_ranges_by_model'][0]:
                alpha_helix_mask[start:end] = True

                # if (label > 0).any():
                #     try:
                #         iou = (label * alpha_helix_mask).sum() / label.sum()
                #         if iou < 0.95:
                #             print(iou)
                #             # pass
                #     except:
                #         pass



        #out = self.sc.run_samcc_turbo(pdbpath=pdb_path, outpath='cache', mode='auto-detect', defdata=None,
	#					plot=False, save_df=False, save_pse=False,
#						layer_detect_n=5, max_dist='auto', search_set_n=9)
        



        # print('asd')
        cc_mask = torch.zeros(size=(num_res,), dtype=torch.float, device=self.device)

        # data_socket = get_socket_data(data_struct)
        # data_twister = get_twister_data(data_struct, data_socket)
        #
        # #cc_mask = data_socket['cc_mask_by_model'][0]
        # cc_mask  = data_twister['cc_mask_by_model'][0]
        #
        # warnings.resetwarnings()

        return cc_mask


    def get_data(self, pdb_path, id = None) -> Tuple[np.array, np.array, np.array]:
        return utils.get_data(pdb_path=pdb_path, dssp_path=self.dssp_path, id=id)


