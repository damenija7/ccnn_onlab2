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
    return np.arctan2(y, x)





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

            twister_dat = twister_main(alpha_carbon_coords, cc_mask, cc_alpha_helix_ranges)
            model_mask |= twister_dat['cc_mask']

        results.append(model_mask)




    return results[0]


def twister_main(alpha_carbon_coords: np.ndarray, cc_mask: np.ndarray, alpha_helix_ranges: List[Tuple[int, int]]) -> np.ndarray:
    num_residues = len(alpha_carbon_coords)

    alpha_helix_ranges = np.array(alpha_helix_ranges, dtype=np.int64)
    alpha_helix_mask = cc_mask

    # atom positioncoords
    A = alpha_carbon_coords

    # (N+2, 3)

    # O = twister_get_O(A)
    O = twister_get_O_alt(A)

    O_padded = np.pad(O, pad_width=((1, 1,), (0,0)), constant_values=-float('inf'))
    O_next = O_padded[2:]
    O_prev = O_padded[0:-2]

    # a_rad: local alpha helical radius per residue
    # a_rise: alpha helical rise per residue
    a_rad = norm(A - O, axis=-1)
    a_rise = (norm(O - O_prev, axis=-1) + norm(O_next - O, axis=-1)) / 2.0


    # alpha helical phase yield per residue
    delta_phi_n = np.zeros_like(a_rad)
    # TODO use parallel alg
    for i in range(1, len(delta_phi_n) -1):
        delta_phi_n[i] = (new_dihedral(A[i-1], O[i-1], O[i], A[i]) + new_dihedral(A[i], O[i], O[i+1], A[i+1])) / 2.0

    pitch = a_rise * 2 * np.pi / (delta_phi_n)

    res_tur = np.pi / delta_phi_n

    max_chain_len = max(range[1] - range[0] for range in alpha_helix_ranges)

    C, C_num_chains = get_C(O, alpha_carbon_coords, alpha_helix_ranges, max_chain_len)

    # local coiled coil radius
    cc_rad = np.zeros_like(C[:, 0])

    for range_start, range_end in alpha_helix_ranges:
        helix_len = range_end - range_start
        cc_rad[:helix_len] += norm(C[:helix_len] - O[range_start:range_end], axis=-1)

    cc_rad /= (C_num_chains +(C_num_chains==0)).squeeze()

    # coiled coil rise per residue
    cc_rise = np.zeros_like(C[:, 0])
    cc_rise[1:-1] = (norm(C[1:-1] - C[0:-2], axis=-1) + norm(C[2:] - C[1:-1], axis=-1)) / 2.0

    alpha = get_crick_phases(A, C, O, alpha_helix_mask, alpha_helix_ranges)
    residue_assignment = get_residue_assignments(alpha)

    return {'cc_mask': torch.tensor([0 if assignment == '0' else 1 for assignment in residue_assignment], dtype=torch.bool),
            'residue_assignment': residue_assignment}





    O_by_chain = []
    for (helix_start, helix_end) in alpha_helix_ranges:
        O_by_chain.append(O[helix_start, helix_end])


def get_C(O, alpha_carbon_coords, alpha_helix_ranges, max_chain_len):
    C = np.zeros(shape=(max_chain_len, 3), dtype=alpha_carbon_coords.dtype)
    C_num_chains = np.zeros_like(C[:, 0], dtype=np.int64)
    for range_start, range_end in alpha_helix_ranges:
        helix_len = range_end - range_start
        C[:helix_len] += O[range_start:range_end]
        C_num_chains[:helix_len] += 1
    C_num_chains = np.expand_dims(C_num_chains, -1)
    C /= (C_num_chains + (C_num_chains == 0))
    return C, C_num_chains


def get_residue_assignments(alpha):
    residue_assignment = ['0' for _ in range(len(alpha))]
    res_idx = 0
    while res_idx < len(alpha) - 2:
        res_idx_incr = 3

        assignment = '0'
        if alpha[res_idx] > 0 and alpha[res_idx - 1] < 0 and abs(alpha[res_idx - 1]) > abs(alpha[res_idx]):
            assignment = 'a'
        elif alpha[res_idx] < 0 and alpha[res_idx + 1] > 0 and abs(alpha[res_idx]) < abs(alpha[res_idx + 1]):
            assignment = 'd'

        if assignment == 'a':
            try:

                residue_assignment[res_idx] = assignment
                residue_assignment[res_idx + 1] = 'b'
                residue_assignment[res_idx + 2] = 'c'
            except:
                pass

        elif assignment == 'd':
            res_idx_incr = 4
            try:
                residue_assignment[res_idx + 1] = 'e'
                residue_assignment[res_idx + 2] = 'f'
                residue_assignment[res_idx + 3] = 'g'
            except:
                pass

        res_idx += res_idx_incr
    return residue_assignment


def get_crick_phases(A, C, O, alpha_helix_mask, alpha_helix_ranges):
    # crick phase
    # Angle between (O_n -> C_n) and O_n -> A_n
    crick_first = np.zeros_like(O)
    crick_second = np.zeros_like(O)
    for ah_start, ah_end in alpha_helix_ranges:
        crick_first[ah_start:ah_end] = C[:(ah_end - ah_start)] - O[ah_start:ah_end]
        crick_second[ah_start:ah_end] = A[ah_start:ah_end] - O[ah_start:ah_end]


    alpha = crick_phase = angle_between_rad(crick_first, crick_second)
    alpha[~alpha_helix_mask] = -float('inf')
    alpha = alpha * (180 / np.pi)
    return alpha


def angle_between_rad(a, b):
    return np.arccos(dot(a/norm(a,axis=-1)[:,None], (b/norm(b,axis=-1)[:,None]).transpose()).diagonal())


def twister_get_O_alt(A):
    A_padded = np.pad(A, pad_width=((2, 2), (0, 0)))
    current_start, current_end = 2, -2
    A_prev = A_padded[current_start - 1:current_end - 1]
    A_current = A
    A_next = A_padded[current_start + 1:current_end + 1]

    I = get_bisection_angle(A_current, A_next, A_prev)
    # (N+1, 3)
    I_padded = np.pad(I, pad_width=((0, 1), (0, 0)))
    # (N, 3)
    I_next = I_padded[1:]


    A_coeff = np.zeros(shape=(A.shape[0], 3,3), dtype=np.float64)
    B_coeff = np.zeros(shape=(A.shape[0], 3), dtype=np.float64)

    # O_n = A_n +I_n * x- > x var
    A_coeff[:, :, 0] = I
    # O_n+1 = - (A_n+1 +I_n+1 * y)- > y var
    A_coeff[:, :, 1] = -I_next
    A_coeff[:, :, 2] = -np.cross(I, I_next)

    B_coeff = -(A - A_next)

    x = np.full(shape=(A.shape[0],), dtype=np.float64, fill_value=-float('inf'))
    y = x.copy()

    for res_idx in range(A.shape[0]):
        try:
            x[res_idx], y[res_idx], _ = np.linalg.solve(A_coeff[res_idx], B_coeff[res_idx])
        except Exception as e:
            pass

    #x_y_t = np.linalg.solve(A_coeff, B_coeff)
    #x = x_y_t[:, 0]
    #y = x_y_t[:, 0]

    O_current = A_current + x[:, None] * I
    O_next = A_next + y[:, None] * I



    O = np.zeros_like(O_current)
    O[0] = O[-1] = -float('inf')
    O[2:-2] = (O_current[2:-2] + O_next[1:-3]) / 2
    # before last only has backwards calc
    O[-2] = O_next[-1]
    # after first only has forwards calc
    O[1] = O_current[1]
    return O


def get_bisection_angle(A_current, A_next, A_prev):
    I_1 = (A_prev - A_current)
    I_1 /= norm(I_1, axis=-1)[:, None]
    I_2 = (A_next - A_current)
    I_2 /= norm(I_2, axis=-1)[:, None]
    I = (I_2 + I_1)
    I /= norm(I, axis=-1)[:, None]
    I[0] = I[-1] = float('-inf')
    return I


def twister_get_O(A):
    A_padded = np.pad(A, pad_width=((2, 2), (0, 0)))
    current_start, current_end = 2, -2
    A_prev = A_padded[current_start - 1:current_end - 1]
    A_current = A
    A_next = A_padded[current_start + 1:current_end + 1]

    # bisections
    # (N, 3)
    I = get_bisection_angle(A_current, A_next, A_prev)


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
