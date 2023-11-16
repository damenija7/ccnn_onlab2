import warnings
from typing import List, Tuple

import numpy as np
import torch
from numpy import dot
from numpy.linalg import norm

from utils.struct_prediction_helper import get_parallel_state_by_chain


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
    alpha_helix_ranges_by_model = data_struct['alpha_helix_ranges_by_model']
    alpha_carbon_coords_by_model = data_struct['alpha_carbon_coords_by_model']
    coiled_coils_by_model = data_socket['cc_by_model']

    results = {}


    for model_idx, (alpha_helix_ranges, coiled_coils, alpha_carbon_coords) in enumerate(zip(alpha_helix_ranges_by_model, coiled_coils_by_model, alpha_carbon_coords_by_model)):
        num_residues = alpha_carbon_coords.shape[0]

        cc_mask = torch.zeros(size=(num_residues,), dtype=torch.bool)
        residue_assignment = ['0' for _ in range(num_residues)]

        for coiled_coil in coiled_coils:
            ah_mask = np.zeros(shape=(num_residues,), dtype=np.bool_)
            cc_alpha_helix_ranges = []
            for alpha_helix_idx in coiled_coil:
                alpha_helix_range = alpha_helix_ranges[alpha_helix_idx]
                ah_mask[alpha_helix_range[0]:alpha_helix_range[1]] = True
                cc_alpha_helix_ranges.append(alpha_helix_range)

            twister_dat = twister_main(alpha_carbon_coords, ah_mask, cc_alpha_helix_ranges)
            cc_mask |= twister_dat['cc_mask']

            for idx, res in enumerate(twister_dat["residue_assignment"]):
                if res != '0' and residue_assignment[idx] == '0':
                    residue_assignment[idx] = res

        results.setdefault('cc_mask_by_model', []).append(cc_mask)
        results.setdefault('residue_assignment_by_model', []).append(residue_assignment)




    return results


def twister_main(alpha_carbon_coords: np.ndarray, alpha_helix_mask: np.ndarray, alpha_helix_ranges: List[Tuple[int, int]]) -> np.ndarray:
    num_residues = len(alpha_carbon_coords)

    alpha_helix_ranges = np.array(alpha_helix_ranges, dtype=np.int64)
    parallel_state_by_chain = get_parallel_state_by_chain(alpha_carbon_coords, alpha_helix_ranges)

    # atom positioncoords
    A = alpha_carbon_coords.astype(np.float64)
    O = np.full_like(A, fill_value=float('-inf'))
    O[1:-1] = twister_get_O_alt(A, 1, A.shape[0]-1, 1)


    #for parallel_state, (ah_start, ah_end) in zip(parallel_state_by_chain, alpha_helix_ranges):
    #    if parallel_state < 0 :
    #        O[ah_start:ah_end] = O_n = twister_get_O_alt(A, ah_start, ah_end, parallel_state)

    O_by_chain = [O[start:end] if parallel_state >= 0 else O[start:end][::-1] for parallel_state, (start, end) in zip(parallel_state_by_chain, alpha_helix_ranges)]
    C, C_num_chains = get_C(O, alpha_carbon_coords, alpha_helix_ranges, parallel_state_by_chain)
    C_prev, C_next = get_prev_next(C)
    O_prev, O_next = get_prev_next(O)

    # alpha helical phase yield per residue
    delta_phi_n = np.zeros_like(O[:, 0])
    # a_rad: local alpha helical radius per residue
    # a_rise: alpha helical rise per residue
    #a_rad = norm(O - A,axis=-1)
    #a_rise = (norm(O - O_prev, axis=-1) + norm(O_next - O, axis=-1)) / 2.0




    # local coiled coil radius
    cc_rad = np.zeros_like(C[:, 0])
    for parallel_state, (range_start, range_end) in zip(parallel_state_by_chain, alpha_helix_ranges):
        helix_len = range_end - range_start
        if parallel_state > 0:
            cc_rad[:helix_len] += norm(C[:helix_len] - O[range_start:range_end], axis=-1)
        else:
            cc_rad[:helix_len] += norm(C[:helix_len] - O[range_start:range_end][::-1], axis=-1)

    cc_rad /= (C_num_chains +(C_num_chains==0)).squeeze()

    # coiled coil rise per residue
    cc_rise = np.zeros_like(C[:, 0])
    cc_rise[1:-1] = (norm(C[1:-1] - C[0:-2], axis=-1) + norm(C[2:] - C[1:-1], axis=-1)) / 2.0

    alpha = get_crick_phases(A=A, C=C, C_next=C_next, C_prev=C_prev, O=O, O_next=O_next, alpha_helix_mask=alpha_helix_mask, alpha_helix_ranges=alpha_helix_ranges,
                             parallel_state_by_chain=parallel_state_by_chain)
    residue_assignment = get_residue_assignments(alpha)

    return {'cc_mask': torch.tensor([0 if assignment == '0' else 1 for assignment in residue_assignment], dtype=torch.bool),
            'residue_assignment': residue_assignment}





    O_by_chain = []
    for (helix_start, helix_end) in alpha_helix_ranges:
        O_by_chain.append(O[helix_start, helix_end])


def get_prev_next(mat):
    mat_padded = np.pad(mat, ((1, 1), (0, 0)), constant_values=float('-inf'))
    mat_prev = mat_padded[0:-2]
    mat_next = mat_padded[2:]
    return mat_prev, mat_next


def get_C(O, alpha_carbon_coords, alpha_helix_ranges, parallel_states_by_chain):
    max_chain_len = max((end-start) for start, end in alpha_helix_ranges)

    C = np.zeros(shape=(max_chain_len, 3), dtype=alpha_carbon_coords.dtype)
    C_num_chains = np.zeros_like(C[:, 0], dtype=np.int64)
    for parallel_state, (range_start, range_end) in zip(parallel_states_by_chain, alpha_helix_ranges):
        helix_len = range_end - range_start
        if parallel_state > 0:
            C[:helix_len] += O[range_start:range_end]
        else:
            C[:helix_len] += (O[range_start:range_end][::-1])
        C_num_chains[:helix_len] += 1
    C_num_chains = np.expand_dims(C_num_chains, -1)
    C /= (C_num_chains + (C_num_chains == 0))
    return C, C_num_chains


def get_residue_assignments(alpha):
    residue_assignment = ['0' for _ in range(len(alpha))]
    res_idx = 1
    while res_idx < len(alpha) - 1:
        res_idx_incr = 1

        alpha_prev, alpha_curr, alpha_next = alpha[res_idx-1], alpha[res_idx], alpha[res_idx+1]

        assignment = '0'
        if alpha_prev < 0 and alpha_curr > 0 and abs(alpha_prev) > abs(alpha_curr):
            assignment = 'a'
        elif alpha_curr < 0 and alpha_next > 0 and abs(alpha_curr) < abs(alpha_next):
            assignment = 'd'

        if assignment == 'a':
            res_idx_incr = 3
            try:

                residue_assignment[res_idx] = assignment
                residue_assignment[res_idx + 1] = 'b'
                residue_assignment[res_idx + 2] = 'c'
            except:
                pass

        elif assignment == 'd':
            res_idx_incr = 4
            try:
                residue_assignment[res_idx] = assignment
                residue_assignment[res_idx + 1] = 'e'
                residue_assignment[res_idx + 2] = 'f'
                residue_assignment[res_idx + 3] = 'g'
            except:
                pass

        res_idx += res_idx_incr
    return residue_assignment


def get_crick_phases_alt(A, C, O, O_next, alpha_helix_mask, alpha_helix_ranges):
    """Calculates Crick angles for the chain."""

    def ClosestPointOnLine(a, b, p):
        ap = p - a
        ab = b - a
        result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return result

    def angle(v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    crick = np.zeros_like(A[:,0])

    O_all, O_next_all = O, O_next
    C_all = C
    C_next_all = np.pad(C_all, pad_width=((0,1),(0,0)))[1:]

    for ah_start, ah_end in alpha_helix_ranges:
        for rel_pos, abs_pos in enumerate(range(ah_start, ah_end)):
            isFirstLayer = abs_pos == 1
            isLastLayer = abs_pos == len(A) - 2


            C = C_all[rel_pos]
            C_next = C_next_all[rel_pos]
            Ca = A[abs_pos]
            O = O_all[abs_pos]
            O_next = O_next_all[abs_pos]

            # Project Ca onto the helix axis (O_new)
            O_new = ClosestPointOnLine(O, O_next, Ca)

            # Define a plane perpendicular to the helix axis
            # and find intersection between this plane and the bundle axis (C_new)
            n = O - O_new
            V0 = O_new
            w = C - O_new
            u = C_next - C
            N = -np.dot(n, w)
            D = np.dot(n, u)
            sI = N / D
            C_new = C + sI * u

            # Define sign of the Crick angle
            mixed = np.dot(np.cross(O_next - O_new, Ca - O_new), C_new - O_new)

            if mixed < 0:
                sign = 1
            else:
                sign = -1

            if not isFirstLayer: sign = sign * -1

            crick[abs_pos] = np.degrees(angle(Ca - O_new, C_new - O_new)) * sign

    return crick



def get_crick_phases(A, C, C_next, C_prev, O, O_next, alpha_helix_mask, alpha_helix_ranges, parallel_state_by_chain):

    # crick phase
    # Angle between (O_n -> C_n) and O_n -> A_n
    crick_first = np.zeros_like(O)
    crick_second = np.zeros_like(O)
    mixed = np.zeros_like(O[:,0])
    for (ah_start, ah_end), parallel in zip(alpha_helix_ranges, parallel_state_by_chain):
        C_ah = C[:(ah_end - ah_start)]
        C_prev_ah = C_prev[:(ah_end-ah_start)]
        C_next_ah = C_next[:(ah_end-ah_start)]
        A_ah = A[ah_start:ah_end]
        O_ah = O[ah_start:ah_end]
        if parallel < 0:
            A_ah = A_ah[::-1]
            O_ah = O_ah[::-1]

        crick_first[ah_start:ah_end] = crick_first_ah = C_ah - O_ah
        crick_second[ah_start:ah_end] = crick_second_ah = A_ah - O_ah

        #O_O_next_ah = O_next[ah_start:ah_end] - O_ah

        # mixed[ah_start:ah_end] = np.dot(np.cross(O_O_next_ah, crick_second[ah_start:ah_end]), crick_first[ah_start:ah_end].transpose()).diagonal()


        c_next_vec_ah = C_ah - C_prev_ah
        c_next_vec_ah[0] = C_next_ah[0] - C_ah[0]

        mixed[ah_start:ah_end] = np.sign(dot(np.cross(crick_first_ah, crick_second_ah), c_next_vec_ah.transpose()).diagonal())



    alpha = crick_phase = angle_between_rad(crick_first, crick_second)
    alpha = alpha * (180 / np.pi) * mixed
    alpha[~alpha_helix_mask] = -float('inf')



    return alpha


def angle_between_rad(a, b):
    a_len, b_len = norm(a, axis=-1), norm(b, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.arccos(dot(a, b.transpose()).diagonal() / a_len / b_len)


def twister_get_O_alt(A, ah_start, ah_end, parallel_state):
    A_orig = A


    A_orig_padded = np.pad(A_orig, pad_width=((1, 1), (0, 0)))

    A = A[ah_start:ah_end]
    A_padded = A_orig_padded[ah_start:ah_end+2]


    if parallel_state < 0:
        A, A_padded = A[::-1], A_padded[::-1]



    A_prev, A_next = A_padded[:-2], A_padded[2:]
    A_current = A

    I = get_bisection_angle(A_current=A_current, A_next=A_next, A_prev=A_prev)
    I_prev, I_next = get_prev_next(I)


    A_coeff = np.zeros(shape=(A.shape[0], 3,3), dtype=np.float64)
    # B_coeff = np.zeros(shape=(A.shape[0], 3), dtype=np.float64)
    B_coeff = A_next - A

    # O_n = A_n +I_n * x- > x var
    A_coeff[:, :, 0] = I
    # O_n+1 = - (A_n+1 +I_n+1 * y)- > y var
    A_coeff[:, :, 1] = -I_next
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        A_coeff[:, :, 2] = np.cross(I, I_next)



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
    O_next = A_next + y[:, None] * I_next

    O = O_combine(O_current, O_next)

    if parallel_state < 0:
        return O[::-1]

    return O


def O_combine(O_current, O_next):
    O = np.full_like(O_current, fill_value=float('-inf'))
    O[1:-1] = (O_current[1:-1] + O_next[0:-2]) / 2

    O[0] = O_current[0]

    if not np.any(np.isnan(O_current[-1]) | np.isinf(O_current[-1])):
        O[-1] = (O_current[-1] + O_next[-2]) / 2.0
    else:
        O[-1] = O_next[-2]

    return O


def get_bisection_angle(A_current, A_next, A_prev):
    I_1 = (A_prev - A_current)
    I_1 /= norm(I_1, axis=-1)[:, None]
    I_2 = (A_next - A_current)
    I_2 /= norm(I_2, axis=-1)[:, None]
    I = (I_2 + I_1)
    I /= norm(I, axis=-1)[:, None]
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
    O_next = A_next + y * I_next
    O = np.zeros_like(O_current)
    O[0] = O[-1] = -float('inf')
    O[2:-2] = (O_current[2:-2] + O_next[1:-3]) / 2
    # before last only has backwards calc
    O[-2] = O_next[-3]
    # after first only has forwards calc
    O[1] = O_current[1]
    return O
