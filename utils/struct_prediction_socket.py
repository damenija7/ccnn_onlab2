import queue
from collections import Counter
from typing import List, Tuple

import networkx
import numpy as np
from numpy.linalg import norm
from numpy import dot

def socket_helix_interactions(alpha_helix_range_1, alpha_helix_range_2, socket_center_coords, packing_cutoff = 7.0):
    alpha_helix_coords_1 = socket_center_coords[alpha_helix_range_1[0]:alpha_helix_range_1[1]]
    alpha_helix_coords_2 = socket_center_coords[alpha_helix_range_2[0]:alpha_helix_range_2[1]]

    num_res_1, num_res_2 = alpha_helix_coords_1.shape[0], alpha_helix_coords_2.shape[0]
    # knob, hole matrix
    # (i,j) ->
    contacts_matrix = np.zeros(shape=(num_res_1, num_res_2), dtype=np.int64)

    for i in range(alpha_helix_coords_1.shape[0]):
        for j in range(alpha_helix_coords_2.shape[0]):
            dist = norm(alpha_helix_coords_1[i] - alpha_helix_coords_2[j])
            if dist <= packing_cutoff:
                contacts_matrix[i,j] += 1


    is_knob_1 = contacts_matrix.sum(axis=-1) >= 4
    is_knob_2 = contacts_matrix.sum(axis=0) >= 4

    knob_1_hole_2 = np.zeros_like(contacts_matrix, dtype=np.bool_)
    knob_2_hole_1 = np.transpose(knob_1_hole_2).copy()

    for knob_1 in np.flatnonzero(is_knob_1):
        knob_1_coord = alpha_helix_coords_1[knob_1]
        hole_2_indices = np.flatnonzero(contacts_matrix[knob_1] > 0)
        hole_2_coords = alpha_helix_coords_2[hole_2_indices]

        distances = np.linalg.norm(hole_2_coords[:, :] - knob_1_coord[None, :], axis=-1)
        closest_4_indices = np.argsort(distances)[:4]

        knob_1_hole_2[knob_1, hole_2_indices[closest_4_indices]] = True


    for knob_2 in np.flatnonzero(is_knob_2):
        knob_2_coord = alpha_helix_coords_2[knob_2]
        hole_1_indices = np.flatnonzero(contacts_matrix[:, knob_2] > 0)
        hole_1_coords = alpha_helix_coords_1[hole_1_indices]

        distances = np.linalg.norm(hole_1_coords[:, :] - knob_2_coord[None, :], axis=-1)
        closest_4_indices = np.argsort(distances)[:4]

        knob_2_hole_1[knob_2, hole_1_indices[closest_4_indices]] = True



    return knob_1_hole_2, knob_2_hole_1

def to_visit(current_path, node, visited, visited_path, visited_chain) -> bool:
    return tuple(current_path + [node]) not in visited_path


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
                knob_1_hole_2, knob_2_hole_1 = socket_helix_interactions(alpha_helix_range_i, alpha_helix_range_j, socket_center_coords)

                knob_hole_matrix[alpha_helix_range_i[0]:alpha_helix_range_i[1], alpha_helix_range_j[0]:alpha_helix_range_j[1]] |= knob_1_hole_2
                knob_hole_matrix[alpha_helix_range_j[0]:alpha_helix_range_j[1], alpha_helix_range_i[0]:alpha_helix_range_i[1]] |= knob_2_hole_1

        # find cycles
        from networkx import simple_cycles

        graph = networkx.from_numpy_array(knob_hole_matrix, create_using=networkx.DiGraph)
        traversal_result = list(simple_cycles(graph))


        # def dfs(start_node):
        #     cycles = set()
        #     cycles_chain = set()
        #     visited = set()
        #     visited_chain = set()
        #     visited_path = set()
        #     s = [start_node]
        #     path = []
        #     path_chain = []
        #
        #     while len(s) != 0:
        #         node = s.pop()
        #         node_chain = index_to_helix_range_index[node]
        #         if to_visit(path, node, visited, visited_path, visited_chain):
        #             # do shit
        #             #
        #             path.append(node)
        #
        #             visited.add(node)
        #             visited_chain.add(node_chain)
        #             visited_path.add(tuple(path))
        #
        #             path_chain.append(node_chain)
        #             num_neighbors_added = 0
        #             for neighbor in np.flatnonzero(knob_hole_matrix[node]):
        #                 neighbor_chain = index_to_helix_range_index[neighbor]
        #                 if to_visit(path, neighbor, visited, visited_path, visited_chain):
        #                     s.append(neighbor)
        #                     num_neighbors_added += 1
        #
        #                 if neighbor in path:
        #                     # neighbor creates cycle
        #                     cycles.add(tuple([*path[path.index(neighbor):], neighbor]))
        #
        #                 if neighbor_chain in path_chain:
        #                     cycles_chain.add(tuple([*path[path_chain.index(neighbor_chain):], neighbor]))
        #
        #             if num_neighbors_added == 0:
        #                 path.pop()
        #                 path_chain.pop()
        #
        #     return cycles, cycles_chain, visited, visited_path
        #
        # cycles, cycles_chain = set(), set()
        # visited = set()
        # visited_path = set()
        #
        # indices_visit = np.flatnonzero(knob_hole_matrix.sum(axis=-1))
        # for idx_i in indices_visit:
        #
        #     cycles_i, cycles_chain_i, visited_i, visited_path_i = dfs(idx_i)
        #
        #     visited |= visited_i
        #     visited_path |= visited_path_i
        #     cycles |= cycles_i
        #     cycles_chain |= cycles_chain_i
        #
        #
        # traversal_result = cycles_chain

        coiled_coils = set()

        num_alpha_helices_involved = [len(Counter([index_to_helix_range_index[cycle_i] for cycle_i in cycle]).keys()) for cycle in traversal_result]
        traversal_result = [x[1] for x in sorted(enumerate(traversal_result), key=lambda i: -num_alpha_helices_involved[i[0]])]
        for cycle_graph in traversal_result:

            alpha_helices_involved = sorted(Counter([index_to_helix_range_index[cycle_i] for cycle_i in cycle_graph]).keys())

            traversal = np.zeros(shape=(len(alpha_helix_ranges), len(alpha_helix_ranges)))

            for cycle_idx in range(len(cycle_graph)):
                cycle_idx_next = (cycle_idx + 1) % len(cycle_graph)

                cycle_idx, cycle_idx_next = cycle_graph[cycle_idx], cycle_graph[cycle_idx_next]

                ah_idx, ah_idx_next = index_to_helix_range_index[cycle_idx], index_to_helix_range_index[cycle_idx_next]

                traversal[ah_idx, ah_idx_next] += 1

            alpha_helix_order = np.floor((traversal + traversal.transpose()).sum(axis=-1) / 2).astype(np.int64)



            if ( len(alpha_helices_involved) <= 2 and np.all(alpha_helix_order[alpha_helix_order > 0] >= 2) ) \
                    or len(alpha_helices_involved) > 3:
                alpha_helix_involved_per_res = frozenset(alpha_helices_involved)
                if not any(alpha_helix_involved_per_res <= cc for cc in coiled_coils):
                    coiled_coils.add(alpha_helix_involved_per_res)
        coiled_coils_by_model.append([list(cc) for cc in coiled_coils])


    return {'coiled_coils_by_model': coiled_coils_by_model}


def socket_data_to_samcc(data_struct, data_socket):
    alpha_helix_ranges_by_model: List[List[Tuple[int, int]]] = data_struct['alpha_helix_ranges_by_model']
    socket_center_coords_by_model: List[np.ndarray] = data_struct['socket_center_coords_by_model']




