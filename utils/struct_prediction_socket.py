from typing import List, Tuple

import numpy as np
from numpy.linalg import norm
from numpy import dot

def socket_has_interaction(alpha_helix_range_1, alpha_helix_range_2, socket_center_coords) -> bool:
    alpha_helix_coords_1 = socket_center_coords[alpha_helix_range_1[0]:alpha_helix_range_1[1]]
    alpha_helix_coords_2 = socket_center_coords[alpha_helix_range_2[0]:alpha_helix_range_2[1]]
    is_knob_1 = np.zeros_like(alpha_helix_coords_1[:,0], dtype=np.bool_)
    is_knob_2 = np.zeros_like(alpha_helix_coords_2[:,0], dtype=np.bool_)

    packing_cutoff = 7.0

    for res_i in range(len(is_knob_1)):
        contacts = np.zeros_like(is_knob_2)
        for res_j in range(len(is_knob_2)):
            contacts[res_j] = norm(alpha_helix_coords_1[res_i] - alpha_helix_coords_2[res_j]) <= packing_cutoff




def get_socket_data(data_struct):
    # alpha_helix_mask_by_model = data_struct['alpha_helix_mask_by_model']
    # alpha_carbon_coords_by_model
    alpha_helix_ranges_by_model: List[List[Tuple[int, int]]] = data_struct['alpha_helix_ranges_by_model']
    socket_center_coords_by_model: List[np.ndarray] = data_struct['socket_center_coords_by_model']

    for model_idx, alpha_helix_ranges in enumerate(alpha_helix_ranges_by_model):
        socket_center_coords = socket_center_coords_by_model[model_idx]
        has_interaction = np.zeros(shape=(len(alpha_helix_ranges), len(alpha_helix_ranges)), dtype=np.bool_)

        for i in range(len(alpha_helix_ranges-1)):
            alpha_helix_range_i = alpha_helix_ranges[i]
            for j in range(i+1, len(alpha_helix_ranges)):
                alpha_helix_range_j = alpha_helix_ranges[j]
                has_interaction[i,j] = has_interaction[j,i] = socket_has_interaction(alpha_helix_range_i, alpha_helix_range_j, socket_center_coords)



