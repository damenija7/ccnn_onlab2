import os.path
import re
import subprocess
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

        self.dssp_path =  'mkdssp'
        self.socket_path = 'socket2'
        self.pcasso_path = 'pcasso'
        self.sc = SamCC(bin_paths={'dssp':self.dssp_path, 'socket':self.socket_path})


        self.dummy_header = 'HEADER    HYDROLASE                               26-SEP-13   3WIW              '

        self.tmp_path = os.getcwd() + '/cache/'
        self.tmp_dssp = self.tmp_path + 'tmp_ccnn_damenija7.dssp'
        self.tmp_dssp_extended = self.tmp_path + 'tmp_ccnn_extended_damenja7.dssp'
        self.tmp_socket = self.tmp_path + 'tmp_ccnn_damenija7.socket'
        self.tmp_socket_extended = self.tmp_path + 'tmp_ccnn_extended_damenija7.socket'

        if not os.path.isdir(self.tmp_path):
            os.makedirs(self.tmp_path)


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


            #output = self.get_output_alt(pdb_path = pdb_path, samcc_path = out_path, label=labels[seq_idx] if labels is not None else None)
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



        out = self.sc.run_samcc_turbo(pdbpath=pdb_path, outpath='cache', mode='auto-detect', defdata=None,
						plot=False, save_df=False, save_pse=False,
						layer_detect_n=5, max_dist='auto', search_set_n=9)
        
        residue_indices = sorted(self.get_residue_indices_set(pdb_path))
        res_idx_dic = {res_idx:i for i, res_idx in enumerate(residue_indices)}

        assignments = ['0' for _ in range(num_res)]
        cc_mask = torch.zeros(size=(num_res,), dtype=torch.float, device=self.device)

        
        for bundle in out:
            for chain in bundle.chains:
                for assignment, res in zip(chain.positions_samcc, chain.res):
                    idx = res_idx_dic[res.res.id[1]]
                    cc_mask[idx] = True
                    assignments[idx] = assignment

        



        # print('asd')
        

        #data_struct = get_data_struct(pdb_path=pdb_path, dssp_path=self.dssp_path, pcasso_path=self.pcasso_path, id=None)
        #data_socket = get_socket_data(data_struct)
        # data_twister = get_twister_data(data_struct, data_socket)
        #
        # #cc_mask = data_socket['cc_mask_by_model'][0]
        # cc_mask  = data_twister['cc_mask_by_model'][0]
        #
        # warnings.resetwarnings()

        return cc_mask


    def get_data(self, pdb_path, id = None) -> Tuple[np.array, np.array, np.array]:
        return utils.get_data(pdb_path=pdb_path, dssp_path=self.dssp_path, id=id)


















    def get_output_alt(self, pdb_path, label: Optional[torch.Tensor] = None, **kwargs):
        # get indices
        #
        residue_indices = self.get_residue_indices_set(pdb_path)
        residuce_indices = sorted(residue_indices)
        num_res: int = len(residue_indices)

        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

        dssp_cmd = [self.dssp_path, pdb_path, self.tmp_dssp]
        socket_cmd = [self.socket_path, '-f', pdb_path, '-s', self.tmp_dssp, '-o', self.tmp_socket]
        socket_cmd_extended = [self.socket_path,'-i' , '    1' ,'-f', pdb_path, '-s', self.tmp_dssp, '-o', self.tmp_socket_extended]

        dssp_res = subprocess.run(dssp_cmd, shell=False, capture_output=True)
        if dssp_res.returncode == 1:
            pass

        # ah_ranges, ah_ranges_extended = self.extend_alpha_helix_ranges(extension=1)

        # if label is not None and label.sum() > 0.0:
        #     ah_mask = torch.zeros_like(label)
        #     for ah_start, ah_end in ah_ranges:
        #         if ah_end != label.shape[0] - 1:
        #             ah_mask[ah_start:ah_end] = True
        #         else:
        #             ah_mask[ah_start:] = True

        #     iou = (label * ah_mask).sum()/label.sum()
        #     print(iou.item())

        #with open(self.tmp_dssp, 'r+') as f:
            #dssp_lines = f.readlines()
            #f.writelines(dssp_lines[:136]
        socket_res = subprocess.run(socket_cmd, shell=False, capture_output=True)
        #if socket_res.returncode == 1:
        #    pass
        with open(self.tmp_socket, 'r') as f:
            socket_lines = f.readlines()

        socket_extended_res = subprocess.run(socket_cmd_extended, shell=False, capture_output=True)
        #if socket_extended_res.returncode == 1:
        #    pass
        with open(self.tmp_socket_extended, 'r') as f:
            socket_extended_lines = f.readlines()

        cc_mask = self.get_cc_mask_from_socket(residuce_indices, num_res, socket_lines)
        cc_mask += self.get_cc_mask_from_socket(residuce_indices, num_res, socket_extended_lines)
        cc_mask[cc_mask > 0] = 1.0

        return cc_mask

    def get_residue_indices_set(self, pdb_path):
        residue_indices = set()

        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM') or line.startswith("HETATM"):
                residue_indices.add(int(line.split()[5]))
        return residue_indices

    def get_cc_mask_from_socket(self, residuce_indices, num_res, socket_lines):
        res_idx_to_assignent = {}
        _icode = "iCode=' '"
        for line in socket_lines:
            try:
                icode_index = line.index(_icode)
            except:
                continue
            if 'helix' not in line and icode_index + len(_icode) < len(line) and line[icode_index + len(_icode)] == 'R':
                assignment = line[icode_index + len(_icode) + 1]
                res_idx = int(line.split()[2].split(':')[0])

                res_idx_to_assignent[res_idx] = assignment

        cc_mask = torch.zeros(size=(num_res,), dtype=torch.float)
        assignments = []

        for i, idx in enumerate(residuce_indices):
            assignment = res_idx_to_assignent.get(idx, '0')
            cc_mask[i] = 1.0 if assignment != '0' else 0.0
            assignments.append(assignment)
        return cc_mask


    def extend_alpha_helix_ranges(self, extension=1):
        with open(self.tmp_dssp, 'r') as f:
            lines = f.readlines()


        alpha_helix_ranges, (start_idx, end_idx) = self.get_alpha_helix_ranges(lines)
        alpha_helix_ranges_original = [tuple(a) for a in alpha_helix_ranges]
        num_res = end_idx - start_idx


        for _ in range(extension):
            for i, (ah_start, ah_end) in enumerate(alpha_helix_ranges):
                prev_line_idx = start_idx + ah_start - 1
                next_line_idx = start_idx + ah_end
                if ah_start-1 >= 0 and not self.get_is_helix(lines[prev_line_idx]):
                    lines[prev_line_idx] = self.change_to_helix(lines[prev_line_idx])
                    alpha_helix_ranges[i][0] -= 1
                if ah_end < num_res and not self.get_is_helix(lines[next_line_idx]):
                    lines[next_line_idx] = self.change_to_helix(lines[next_line_idx])
                    alpha_helix_ranges[i][1] += 1

        if extension > 0:
            with open(self.tmp_dssp_extended, 'w') as f:
                f.writelines(lines)

        return alpha_helix_ranges_original, alpha_helix_ranges



    def get_alpha_helix_ranges(self, lines):

        start_idx = None
        end_idx = len(lines) - 1

        for i, line in enumerate(lines):
            if line.strip().startswith(r'#  RESIDUE AA STRUCTURE'):
                start_idx = i+1
                break

        alpha_helix_ranges = [
        ]

        ah_start = None

        for i in range(start_idx, end_idx):
            line = lines[i]
            is_helix = self.get_is_helix(line)

            if is_helix:
                if ah_start is None:
                    ah_start = i
            elif ah_start is not None:
                alpha_helix_ranges.append([ah_start - start_idx, i - start_idx])
                ah_start = None

        if ah_start is not None:
            alpha_helix_ranges.append([ah_start - start_idx, i - start_idx])
            ah_start = None


        return alpha_helix_ranges, (start_idx, end_idx)

    def get_is_helix(self, line):
        is_helix = line.split()[4] == 'H'
        return is_helix


    def change_to_helix(self, line):
        #regexp = r"\s*([\da-zA-Z])\s*"
        #res = [(m.start(0), m.end(0)) for m in re.finditer(regexp, line)]

        #start_idx, end_idx = res[4]

        #changed_line = line[:start_idx] + (' '*(end_idx - start_idx - 1) + 'H') + line[end_idx:]
        changed_line = line[:16] + 'H' + line[17:]

        return changed_line



