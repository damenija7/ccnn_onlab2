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


    def get_data(self, pdb_path, id = None) -> Tuple[np.array, np.array]:
        pdb_parser = Bio.PDB.PDBParser()

        if id == None:
            id = pdb_path.rsplit('/')[0]

        # %%
        struct = pdb_parser.get_structure(file=pdb_path, id=id)
        # %%


        # %%
        models: List[Model] = list(struct.get_models())
        atoms_by_model: List[List[Atom]] = [list(model.get_atoms()) for model in models]
        atom_coords_by_model = [np.stack([atom.get_coord() for atom in atoms]) for atoms in atoms_by_model]
        dssp_by_model: List[DSSP] = [Bio.PDB.DSSP(model, pdb_path, dssp=self.dssp_path) for model in models]
        # %%
        from typing import Tuple

        alpha_helices_by_model: List[List[Tuple]] = []

        for dssp in dssp_by_model:
            alpha_helices = []
            for res_key in dssp.keys():
                res = dssp[res_key]
                # is alpha helix
                if res[2] == 'H':
                    alpha_helices.append(res)

            alpha_helices_by_model.append(alpha_helices)
        # %%

        alpha_helix_mask_by_model = [np.zeros(shape=(len(atom_coords),), dtype=bool) for atom_coords in
                                     atom_coords_by_model]

        for alpha_helix_mask, alpha_helices in zip(alpha_helix_mask_by_model, alpha_helices_by_model):
            for res_info in alpha_helices:
                # 0 <-> RES INDEX
                alpha_helix_mask[res_info[0]] = True
        # %%
        return atom_coords_by_model, alpha_helix_mask_by_model


