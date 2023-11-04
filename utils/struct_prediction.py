import numpy as np
from typing import Tuple, List
import Bio
from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Residue import Residue

from utils.struct_prediction_socket import get_socket_data
from utils.struct_prediction_twister import get_twister_data

test_fname='AF-A0A4W3JAN5-F1-model_v4.pdb'
test_dssp_path='/home/damenija7/Apps/dssp.AppImage'


def get_data_struct(pdb_path, dssp_path, id=None) -> Tuple[np.array, np.array, np.array]:
    pdb_parser = Bio.PDB.PDBParser()

    if id == None:
        id = pdb_path.rsplit('/')[0]

    # %%
    struct = pdb_parser.get_structure(file=pdb_path, id=id)
    # %%

    # %%
    models: List[Model] = list(struct.get_models())
    residues_by_model: List[List[Residue]] = [list(model.get_residues()) for model in models]




    alpha_helix_ranges_by_model, alpha_helix_mask_by_model = get_dssp_info(models, dssp_path, pdb_path)
    alpha_carbon_coords_by_model = [np.stack([res['CA'].coord for res in residues]) for residues in residues_by_model]
    socket_center_coords_by_model = []

    for model_idx, model in enumerate(models):
        residues = residues_by_model[model_idx]

        socket_center_coords = []

        for residue in residues:
            # BASED ON SOCKET
            if residue.resname == 'GLY':
                center = residue['CA'].coord
            else:
                # mean co-ordinate of all the side-chain atoms (excluding hydrogens) from Cb onwards
                atoms = [atom for atom in list(residue.get_atoms()) if len(atom.fullname.strip()) >= 2 and atom.fullname.strip()[1] != 'A']
                center = sum(atom.coord for atom in atoms)/len(atoms)

            socket_center_coords.append(center)

        socket_center_coords = np.stack(socket_center_coords)
        socket_center_coords_by_model.append(socket_center_coords)





    # %%
    return {
        'alpha_helix_mask_by_model': alpha_helix_mask_by_model,
        'alpha_helix_ranges_by_model': alpha_helix_ranges_by_model,
        'alpha_carbon_coords_by_model': alpha_carbon_coords_by_model,
        'socket_center_coords_by_model': socket_center_coords_by_model
    }







def get_dssp_info(models, dssp_path, pdb_path):

    num_residues_by_model: List[int] = [len(list(model.get_residues())) for model in models]

    dssp_by_model: List[DSSP] = [Bio.PDB.DSSP(model, pdb_path, dssp=dssp_path) for model in models]
    # %%
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

    alpha_helix_mask_by_model = [np.zeros(shape=(num_residues,), dtype=bool) for num_residues in
                                 num_residues_by_model]
    for alpha_helix_mask, alpha_helices in zip(alpha_helix_mask_by_model, alpha_helices_by_model):
        for res_info in alpha_helices:
            # 0 <-> RES INDEX ( counting starts from zero, must -= 1)
            alpha_helix_mask[res_info[0] - 1] = True
    alpha_helix_ranges_by_model = []
    for alpha_helix_mask in alpha_helix_mask_by_model:
        alpha_helix_ranges = []

        helix_range_start_idx = None

        for i, is_helix in enumerate(alpha_helix_mask):
            if is_helix and helix_range_start_idx is None:
                helix_range_start_idx = i
            elif not is_helix and helix_range_start_idx is not None:
                alpha_helix_ranges.append((helix_range_start_idx, i))
                helix_range_start_idx = None

        # check end
        if helix_range_start_idx is not None:
            alpha_helix_ranges.append((helix_range_start_idx, len(alpha_helix_mask)))
            helix_range_start_idx = None

        alpha_helix_ranges_by_model.append(alpha_helix_ranges)

    return alpha_helix_ranges_by_model, alpha_helix_mask_by_model



if __name__ == '__main__':
    data_struct = get_data_struct(test_fname, test_dssp_path)
    data_socket = get_socket_data(data_struct)
    data_twister = get_twister_data(data_struct, data_socket)
