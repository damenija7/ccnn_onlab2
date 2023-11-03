import numpy as np
from typing import Tuple, List
import Bio
from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from Bio.PDB.DSSP import DSSP


test_fname='AF-A0A4W3JAN5-F1-model_v4.pdb'
test_dssp_path='/home/damenija7/Apps/dssp.AppImage'


def new_dihedral(p0, p1, p2, p3):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def twister(atom_coords: np.ndarray, alpha_helix_mask: np.ndarray, alpha_helix_ranges: List[Tuple[int, int]]) -> np.ndarray:
    num_residues = len(atom_coords)

    alpha_helix_ranges = np.array(alpha_helix_ranges, dtype=np.int64)

    # atom positioncoords
    A = atom_coords

    # (N+2, 3)
    A_padded = np.pad(A, pad_with=((2, 2), (0,0)))


    current_start, current_end = 2, -2
    A_prev_prev = A_padded[current_start-2:current_end-2]
    A_prev = A_padded[current_start-1:current_end-1]
    A_current = A
    A_next = A_padded[current_start+1:current_end+1]
    A_next_next = A_padded[current_start+2:current_end+2]

    # bisections
    # (N, 3)
    I = (A_prev - A_current) + (A_next - A_current)
    I /= np.linalg.norm(I)

    # (N+1, 3)
    I_padded = np.pad(I, pad_with=((0, 1), (0,0)))
    # (N, 3)
    I_next = I_padded[1:]

    # forward o_n backward o_{n+1}
    # a = I_n dot I_n, b = I_n dot I_n+1
    a = np.dot(I, np.transpose(I))
    b = np.dot(I, np.transpose(I_next))
    # c = (A_{N+1} - A_n) dot I_n
    # d = (A_{N+1} - A_n) dot I_{n+1}
    c = np.dot((A_next - A), np.transpose(I))
    d = np.dot((A_next - A), np.transpose(I_next))

    # O_N = A_n + x I_n
    # O_{N+1} = A_{n+1} + y I{n+1}
    # TODO Verify
    x = (d - (a * c) / b) / (b - a * a / b)
    y = (c - a * x) / b

    O_current = A_current + x * I
    O_next = A_next + y * y * I

    O = np.zeros_like(O_current)

    O[0] = O[-1] = -64
    O[2:-2] = (O_current[2:-2] + O_next[1:-3]) / 2
    # before last only has backwards calc
    O[-2] = O_next[-1]
    # after first only has forwards calc
    O[1] = O_current[1]

    # local alpha helical radius
    r_n = np.linalg.norm(A - O_next, axis=-1)
    # alpha helical rise per residue
    h_n = np.zeros_like(r_n)

    h_n[1:-1] = (np.linalg.norm(O[1:-1] - O[0:-2], axis=-1) + np.linalg.norm(O[2:] - O[1:-1], axis=-1)) / 2.0

    # alpha helical phase yield per residue
    delta_phi_n = np.zeros_like(r_n)

    # TODO use parallel alg
    for i in range(1, len(delta_phi_n) -1):
        delta_phi_n[i] = (new_dihedral(A[i-1], O[i-1], O[i], A[i]) + new_dihedral(A[i], O[i], O[i+1], A[i+1])) / 2.0


    O_by_chain = []
    for (helix_start, helix_end) in alpha_helix_ranges:
        O_by_chain.append(O[helix_start, helix_end])







def get_data(pdb_path, dssp_path, id=None) -> Tuple[np.array, np.array, np.array]:
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
    dssp_by_model: List[DSSP] = [Bio.PDB.DSSP(model, pdb_path, dssp=dssp_path) for model in models]
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
                alpha_helix_ranges.append(helix_range_start_idx, i)
                helix_range_start_idx = None

        # check end
        if helix_range_start_idx is not None:
            alpha_helix_ranges.append(helix_range_start_idx, len(alpha_helix_mask))
            helix_range_start_idx = None

        alpha_helix_ranges_by_model.append(alpha_helix_ranges)


    # %%
    return atom_coords_by_model, alpha_helix_mask_by_model, alpha_helix_ranges_by_model



if __name__ == '__main__':
    get_data(test_fname,  test_dssp_path)