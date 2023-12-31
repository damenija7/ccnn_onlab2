import subprocess
import warnings
from collections import namedtuple

import numpy as np
from typing import Tuple, List
import Bio
import pydssp
import torch
from Bio.PDB.Model import Model
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Residue import Residue

from utils.struct_prediction_socket import get_socket_data
from utils.struct_prediction_twister import get_twister_data


AMINO_ACID = set(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                  'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])


HelixClass = namedtuple('HelixClass',
                        ['start', 'end', 'chain', 'ap', 'cc_id'])




def get_data_struct(pdb_path, dssp_path, pcasso_path, id=None) -> Tuple[np.array, np.array, np.array]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pdb_parser = Bio.PDB.PDBParser()

        if id == None:
            id = pdb_path.rsplit('/')[0]

    # %%
        struct = pdb_parser.get_structure(file=pdb_path, id=id)
    # %%

    # %%
    models: List[Model] = list(struct.get_models())
    residues_by_model: List[List[Residue]] = [list(model.get_residues()) for model in models]



    alpha_helix_ranges_by_model = get_dssp_info(models, dssp_path, pdb_path)
    #alpha_helix_ranges_by_model = get_dssp_info_alt(models, dssp_path, pdb_path)
    #alpha_helix_ranges_by_model = get_dssp_info_pcasso(models=models, pcasso_path=pcasso_path, pdb_path=pdb_path)
    # alpha_carbon_coords_by_model = [np.stack([(res['CA'] if not res['CA'].is_disordered() else res['CA'].disordered_get_list()[0]).coord for res in residues if 'CA' in res]) for residues in residues_by_model]
    alpha_carbon_coords_by_model = [np.stack([(res['CA']).coord for res in residues if 'CA' in res]) for residues in residues_by_model]
    socket_center_coords_by_model = []

    for model_idx, model in enumerate(models):
        residues = residues_by_model[model_idx]

        socket_center_coords = []

        for residue in residues:
            if 'CA' not in residue:
                continue

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
        'alpha_helix_ranges_by_model': alpha_helix_ranges_by_model,
        'alpha_carbon_coords_by_model': alpha_carbon_coords_by_model,
        'socket_center_coords_by_model': socket_center_coords_by_model,
        'models': models
    }


def get_dssp_info_pcasso(models, pcasso_path, pdb_path):
    cmd = [pcasso_path, pdb_path]
    assignments = subprocess.run(cmd,
                                 capture_output=True,
                                 text=True
                                 ).stdout[len(pdb_path)+1:].split()
    return assignments_to_alpha_helices_by_model([assignments])



def get_dssp_info_alt(models, dssp_path, pdb_path):



    #with open(pdb_path, 'r') as pdb_f:
    #    dssp_by_model = [pydssp.assign(torch.tensor(pydssp.read_pdbtext(pdb_f.read())))]

    cmd = [dssp_path, pdb_path]
    run_results = subprocess.run(cmd,
                                 capture_output=True,
                                 text=True
                                 )

    assignments = run_results.stdout

    assignments = assignments.strip('\n')
    assignments = [assignment.split()[4] for assignment in assignments]

    return assignments_to_alpha_helices_by_model([assignments])


def assignments_to_alpha_helices_by_model(dssp_by_model):
    num_residues_by_model = [len(dssp) for dssp in dssp_by_model]
    # %%
    alpha_helices_by_model: List[List[Tuple]] = []
    for dssp in dssp_by_model:
        current_alpha_helix_start = None
        alpha_helices = []
        for res_idx, res in enumerate(dssp):
            # is alpha helix
            if res == 'H':
                if current_alpha_helix_start is None:
                    current_alpha_helix_start = res_idx
            else:
                if current_alpha_helix_start is not None:
                    alpha_helices.append((current_alpha_helix_start, res_idx))
                    current_alpha_helix_start = None

        if current_alpha_helix_start is not None:
            alpha_helices.append((current_alpha_helix_start, len(dssp)))

        alpha_helices_by_model.append(alpha_helices)
    # %%
    return alpha_helices_by_model


def get_dssp_info(models, dssp_path, pdb_path):
    dssp_by_model: List[DSSP] = [Bio.PDB.DSSP(model, pdb_path, dssp=dssp_path) for model in models]

    # num_residues_by_model: List[int] = [len([ residue for residue in model.get_residues() if 'CA' in residue]) for model in models]
    num_residues_by_model = [len(dssp) for dssp in dssp_by_model]

    # %%
    alpha_helices_by_model: List[List[Tuple]] = []

    for dssp in dssp_by_model:
        current_alpha_helix_start = None
        alpha_helices = []
        for res_idx, res_key in enumerate(dssp.keys()):
            res = dssp[res_key]
            # is alpha helix
            if res[2] == 'H':
                if current_alpha_helix_start is None:
                    current_alpha_helix_start = res_idx
            else:
                if current_alpha_helix_start is not None:
                    alpha_helices.append((current_alpha_helix_start, res_idx))
                    current_alpha_helix_start = None

        if current_alpha_helix_start is not None:
            alpha_helices.append((current_alpha_helix_start, len(dssp.keys())))

        alpha_helices_by_model.append(alpha_helices)
    # %%



    return alpha_helices_by_model



#test_fname='AF-A0A4W3JAN5-F1-model_v4.pdb'
test_fname = '2zta.pdb'
# test_fname = '1d7m.pdb'
#test_fname='1fu1.pdb'
test_dssp_path='/home/damenija7/Apps/dssp.AppImage'
test_pcasso_path='pcasso'

if __name__ == '__main__':
    data_struct = get_data_struct(test_fname, test_dssp_path, test_pcasso_path)
    data_socket = get_socket_data(data_struct)
    data_twister = get_twister_data(data_struct, data_socket)

    print(data_twister)


