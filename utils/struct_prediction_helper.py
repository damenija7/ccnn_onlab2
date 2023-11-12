import numpy as np


def get_parallel_state_by_chain(alpha_carbon_coords, alpha_helix_ranges):
    parallel_state_by_chain = np.ones(shape=(len(alpha_helix_ranges)), dtype=np.int8)
    first_chain_dir = alpha_carbon_coords[alpha_helix_ranges[0][1] - 1] - alpha_carbon_coords[alpha_helix_ranges[0][0]]
    for i, (ah_start, ah_end) in enumerate(alpha_helix_ranges[1:]):
        i = i + 1
        chain_dir = alpha_carbon_coords[ah_end - 1] - alpha_carbon_coords[ah_start]

        parallel_state_by_chain[i] = 1 if np.dot(first_chain_dir, chain_dir) >= 0 else -1
    return parallel_state_by_chain
