from typing import List, Tuple

import networkx
import numpy as np
from numpy.linalg import norm
from numpy import dot

def socket_has_interaction(alpha_helix_range_1, alpha_helix_range_2, socket_center_coords, packing_cutoff = 7.0) -> bool:
    alpha_helix_coords_1 = socket_center_coords[alpha_helix_range_1[0]:alpha_helix_range_1[1]]
    alpha_helix_coords_2 = socket_center_coords[alpha_helix_range_2[0]:alpha_helix_range_2[1]]

    num_res_1, num_res_2 = alpha_helix_coords_1.shape[0], alpha_helix_coords_2.shape[0]
    # knob, hole matrix
    # (i,j) -> ith res of helix 1 is knob for hole, jth res of helix 2 ?
    contacts_matrix = np.zeros(shape=(num_res_1, num_res_2), dtype=np.int64)

    num_contacts_1 = np.zeros(shape=(num_res_1,), dtype=np.int64)
    num_contacts_2 = np.zeros(shape=(num_res_2,), dtype=np.int64)

    for i in range(alpha_helix_coords_1.shape[0]):
        for j in range(alpha_helix_coords_2.shape[0]):
            dist = norm(alpha_helix_coords_1[i] - alpha_helix_coords_2[j])
            if dist <= packing_cutoff:
                contacts_matrix[i,j] += 1


    is_knob_1 = contacts_matrix.sum(axis=-1) >= 4
    is_knob_2 = contacts_matrix.sum(axis=0) >= 4

    knob_1_hole_2 = np.zeros_like(contacts_matrix, dtype=np.int64)
    knob_2_hole_1 = np.transpose(knob_1_hole_2).copy()

    if np.any(is_knob_1):
        knob_1_hole_2[is_knob_1] = contacts_matrix[is_knob_1] > 0
    if np.any(is_knob_2):
        knob_2_hole_1[is_knob_2] = np.transpose(contacts_matrix[:, is_knob_2] > 0)



    return knob_1_hole_2, knob_2_hole_1


def fill_is_knob_1(alpha_helix_coords_1, alpha_helix_coords_2, is_knob_1, is_knob_2, packing_cutoff):
    for res_i in range(len(is_knob_1)):
        contacts = np.zeros_like(is_knob_2)
        for res_j in range(len(is_knob_2)):
            dist = norm(alpha_helix_coords_1[res_i] - alpha_helix_coords_2[res_j])
            contacts[res_j] = dist <= packing_cutoff

        is_knob_1[res_i] = contacts.sum() >= 4


def get_socket_data(data_struct):
    # alpha_helix_mask_by_model = data_struct['alpha_helix_mask_by_model']
    # alpha_carbon_coords_by_model
    alpha_helix_ranges_by_model: List[List[Tuple[int, int]]] = data_struct['alpha_helix_ranges_by_model']
    socket_center_coords_by_model: List[np.ndarray] = data_struct['socket_center_coords_by_model']

    coiled_coils_by_model = []

    for model_idx, alpha_helix_ranges in enumerate(alpha_helix_ranges_by_model):
        socket_center_coords = socket_center_coords_by_model[model_idx]
        num_residues = socket_center_coords.shape[0]
        has_interaction = np.zeros(shape=(len(alpha_helix_ranges), len(alpha_helix_ranges)), dtype=np.bool_)

        knob_hole_matrix = np.zeros(shape=(num_residues, num_residues), dtype=np.bool_)
        index_to_helix_range_index = np.full(shape=(num_residues), dtype=np.int64, fill_value=-1)

        for helix_idx, (helix_range_start, helix_range_end) in enumerate(alpha_helix_ranges):
            index_to_helix_range_index[helix_range_start:helix_range_end] = helix_idx



        for i in range(len(alpha_helix_ranges)-1):
            alpha_helix_range_i = alpha_helix_ranges[i]
            for j in range(i+1, len(alpha_helix_ranges)):
                alpha_helix_range_j = alpha_helix_ranges[j]
                knob_1_hole_2, knob_2_hole_1 = socket_has_interaction(alpha_helix_range_i, alpha_helix_range_j, socket_center_coords)

                knob_hole_matrix[alpha_helix_range_i[0]:alpha_helix_range_i[1], alpha_helix_range_j[0]:alpha_helix_range_j[1]] = knob_1_hole_2
                knob_hole_matrix[alpha_helix_range_j[0]:alpha_helix_range_j[1], alpha_helix_range_i[0]:alpha_helix_range_i[1]] = knob_2_hole_1

        # find cycles
        from networkx import simple_cycles

        graph = networkx.from_numpy_array(knob_hole_matrix, create_using=networkx.DiGraph)
        cycles = simple_cycles(graph)

        coiled_coils = set()
        for cycle in cycles:
            cycle = [index_to_helix_range_index[cycle_i] for cycle_i in cycle]
            cycle = tuple(sorted(set(cycle)))
            coiled_coils.add(cycle)

        coiled_coils_by_model.append(list(coiled_coils))


    return {'coiled_coils_by_model': coiled_coils_by_model}




