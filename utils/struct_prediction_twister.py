from typing import List, Tuple

import numpy as np
import torch
from numpy import dot
from numpy.linalg import norm


def new_dihedral(p0, p1, p2, p3):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - dot(b0, b1)*b1
    w = b2 - dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = dot(v, w)
    y = dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))





def get_twister_data(data_struct, data_socket):
    alpha_helix_mask_by_model, alpha_helix_ranges_by_model = data_struct['alpha_helix_mask_by_model'], data_struct['alpha_helix_ranges_by_model']
    alpha_carbon_coords_by_model = data_struct['alpha_carbon_coords_by_model']
    coiled_coils_by_model = data_socket['coiled_coils_by_model']

    results = []


    for model_idx, (alpha_helix_mask, alpha_helix_ranges, coiled_coils, alpha_carbon_coords) in enumerate(zip(alpha_helix_mask_by_model, alpha_helix_ranges_by_model, coiled_coils_by_model, alpha_carbon_coords_by_model)):
        num_residues = alpha_carbon_coords.shape[0]

        model_mask = torch.zeros(size=(num_residues,), dtype=torch.bool)

        for coiled_coil in coiled_coils:
            cc_mask = np.zeros(shape=(num_residues,), dtype=np.bool_)
            cc_alpha_helix_ranges = []
            for alpha_helix_idx in coiled_coil:
                alpha_helix_range = alpha_helix_ranges[alpha_helix_idx]
                cc_mask[alpha_helix_range[0]:alpha_helix_range[1]] = True
                cc_alpha_helix_ranges.append(alpha_helix_range)

            model_mask |= twister_main(alpha_carbon_coords, cc_mask, cc_alpha_helix_ranges)

        results.append(model_mask)




    return results


def twister_main(alpha_carbon_coords: np.ndarray, cc_mask: np.ndarray, alpha_helix_ranges: List[Tuple[int, int]]) -> np.ndarray:
    num_residues = len(alpha_carbon_coords)

    alpha_helix_ranges = np.array(alpha_helix_ranges, dtype=np.int64)

    # atom positioncoords
    A = alpha_carbon_coords

    # (N+2, 3)

    O = twister_get_O(A)
    O_padded = np.pad(O, pad_width=((1, 1,), (0,0)), constant_values=-float('inf'))
    O_next = O_padded[2:]
    O_prev = O_padded[0:-2]

    # local alpha helical radius
    r_n = norm(A - O_next, axis=-1)
    # alpha helical rise per residue
    h_n = np.zeros_like(r_n)

    h_n[1:-1] = (norm(O[1:-1] - O[0:-2], axis=-1) + norm(O[2:] - O[1:-1], axis=-1)) / 2.0

    # alpha helical phase yield per residue
    delta_phi_n = np.zeros_like(r_n)
    # TODO use parallel alg
    for i in range(1, len(delta_phi_n) -1):
        delta_phi_n[i] = (new_dihedral(A[i-1], O[i-1], O[i], A[i]) + new_dihedral(A[i], O[i], O[i+1], A[i+1])) / 2.0

    max_chain_len = max(range[1] - range[0] for range in alpha_helix_ranges)


    C = np.zeros(shape=(max_chain_len, 3), dtype=alpha_carbon_coords.dtype)
    C_num_chains = np.zeros_like(C[:,0], dtype=np.int64)

    for range_start, range_end in alpha_helix_ranges:
        helix_len = range_end - range_start
        C[:helix_len] += O[range_start:range_end]
        C_num_chains[:helix_len] += 1

    C_num_chains = np.expand_dims(C_num_chains, -1)
    C /= (C_num_chains + (C_num_chains==0))

    # local coiled coil radius
    R = np.zeros_like(C[:, 0])

    for range_start, range_end in alpha_helix_ranges:
        helix_len = range_end - range_start
        R[:helix_len] += norm(C[:helix_len] - O[range_start:range_end], axis=-1)

    R /= (C_num_chains +(C_num_chains==0)).squeeze()

    # coiled coil rise per residue
    H = np.zeros_like(C[:, 0])
    H[1:-1] = (norm(C[1:-1] - C[0:-2], axis=-1) + norm(C[2:] - C[1:-1], axis=-1)) / 2.0


    # crick phase
    # Angle between (O_n -> C_n) and O_n -> A_n
    crick_first = C - O
    crick_first /= norm(crick_first, axis=-1)
    crick_second = A - O
    crick_second /= norm(crick_second, axis=-1)
    alpha = crick_phase = np.arccos(dot(crick_first, crick_second))


    residue_assignment = ['0' for _ in range(len(A))]

    for res_idx in range(1, len(A) - 1):
        assigment = '0'
        if alpha[res_idx] > 0 and alpha[res_idx - 1] < 0 and abs(alpha[res_idx-1]) > abs(alpha[res_idx]):
            assignment = 'a'
        elif alpha[res_idx] < 0 and alpha[res_idx + 1] > 0 and abs(alpha[res_idx]) < abs(alpha[res_idx+1]):
            assignment = 'd'

        if assignment == 'a':
            residue_assignment[res_idx] = assignment
            residue_assignment[res_idx+1] = 'b'
            residue_assignment[res_idx+2] = 'c'

        elif assigment == 'd':
            residue_assignment[res_idx + 1] = 'e'
            residue_assignment[res_idx + 2] = 'f'
            residue_assignment[res_idx + 1] = 'g'


    return torch.tensor([0 if assignment == '0' else 1 for assignment in residue_assignment])





    O_by_chain = []
    for (helix_start, helix_end) in alpha_helix_ranges:
        O_by_chain.append(O[helix_start, helix_end])


def twister_get_O(A):
    A_padded = np.pad(A, pad_width=((2, 2), (0, 0)))
    current_start, current_end = 2, -2
    A_prev = A_padded[current_start - 1:current_end - 1]
    A_current = A
    A_next = A_padded[current_start + 1:current_end + 1]

    # bisections
    # (N, 3)
    I = (A_prev - A_current) + (A_next - A_current)
    I /= np.expand_dims(norm(I, axis=-1), -1)
    # (N+1, 3)
    I_padded = np.pad(I, pad_width=((0, 1), (0, 0)))
    # (N, 3)
    I_next = I_padded[1:]
    # forward o_n backward o_{n+1}
    # a = I_n dot I_n
    # b = I_n dot I_n+1
    # c = I_{n+1} dot I{n+1}
    a = np.diag(dot(I, np.transpose(I)))
    b = np.diag(dot(I, np.transpose(I_next)))
    c = np.diag(dot(I_next, np.transpose(I_next)))
    # e = (A_{N+1} - A_n) dot I_n
    # d = (A_{N+1} - A_n) dot I_{n+1}
    d = np.diag(dot((A_next - A), np.transpose(I)))
    e = np.diag(dot((A_next - A), np.transpose(I_next)))
    # O_N = A_n + x I_n
    # O_{N+1} = A_{n+1} + y I{n+1}
    # TODO Verify
    # ax - by = d
    # bx - cy = e
    x = (e - c * d / b) / (b - c * a / b)
    y = (a * x - d) / b
    # numpy broadcasting
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)
    O_current = A_current + x * I
    O_next = A_next + y * I
    O = np.zeros_like(O_current)
    O[0] = O[-1] = -float('inf')
    O[2:-2] = (O_current[2:-2] + O_next[1:-3]) / 2
    # before last only has backwards calc
    O[-2] = O_next[-1]
    # after first only has forwards calc
    O[1] = O_current[1]
    return O
