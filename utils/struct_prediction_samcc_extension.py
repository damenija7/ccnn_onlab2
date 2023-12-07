
def parse_socket(data_struct, data_socket):
    """Parse raw SOCKET data and generate a list of helices

    Arguments:
    socket_data -- raw SOCKET data

    Returns:
    bundles: list of bundles (i.e list of helices: from, to, chain, anti-parallel)
    """

    bundles = []
    alpha_helix_ranges_by_model = data_struct['alpha_helix_ranges_by_model']

    for cc_idx, cc in enumerate(data_socket['coiled_coils_by_model'][0]):
        # detect helices orientation

        bundleDesc = []
        for helix_idx in cc:
            bundleDesc.append([*alpha_helix_ranges_by_model[0][helix_idx], cc_idx, False, cc_idx])

        # if all helices in a bundle are AP then we should
        # switch them all to P (parallel)

        bundles.append(bundleDesc)

    return bundles