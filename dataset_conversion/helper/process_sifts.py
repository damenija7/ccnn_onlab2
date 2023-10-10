from typing import Dict, Tuple

# %%
from xml.etree.ElementTree import Element


def get_pdb_uniprot_mapping_by_chain(
    sifts_entry_xml_path: str,
) -> dict[str, dict[int, tuple[str, int]]]:
    """
    For given PDB Entry, Return Chain ID -> PDB Residue Index -> (Uniprot Entry ID, Uniprot Residue Idx) Dat

    :param sifts_entry_xml_path: SIFTS Entry for PDB ID
    :return: For given PDB Entry, Return Chain ID -> PDB Residue Index -> (Uniprot Entry ID, Uniprot Residue Idx) Dat
    """
    ####
    # output
    # chain -> pdb_res_idx -> (uniprot id, uniprot_res_idx)
    pdb_uniprot_mapping_list_by_chain: Dict[str, Dict[int, Tuple[str, int]]] = {}

    from xml.etree import ElementTree as ET

    # whole xml document to iterate over
    root = ET.iterparse(sifts_entry_xml_path, events=("start", "end"))
    # an xml element
    elem: Element
    # either 'start' (<x>) or 'end' (</x>)
    event: str
    # inside "<chain>" entity if not None
    current_chain_id = None
    # inside "<residue>" entity
    in_residue = False

    resnum_pdb, resnum_uniprot, uniprot_id = None, None, None

    for idx, (event, elem) in enumerate(root):
        # Start/End of chain
        if "entity" in elem.tag:
            if elem.attrib.get("type", "") == "protein":
                if event == "end":
                    current_chain_id = None
                elif event == "start":
                    current_chain_id = elem.attrib["entityId"]

        # if we're not in a chain no need to process anything
        if not current_chain_id:
            continue

        if "residue" in elem.tag and "residueDetail" not in elem.tag:
            if event == "start":
                in_residue = True
                resnum_pdb, resnum_uniprot, uniprot_id = None, None, None
            else:
                if resnum_pdb is not None and resnum_uniprot is not None:
                    # HAVE (PDB IDX, UNIPROT IDX) Pair
                    pdb_uniprot_mapping_list_by_chain.setdefault(current_chain_id, {})[
                        resnum_pdb
                    ] = (uniprot_id, resnum_uniprot)

                    in_residue = False

        if event != "start":
            continue

        if in_residue and "crossRefDb" in elem.tag:
            # 'dbSource': 'PDB', 'dbCoordSys': 'PDBresnum'
            if (
                elem.attrib["dbSource"] == "PDB"
                and elem.attrib["dbCoordSys"] == "PDBresnum"
            ):
                try:
                    resnum_pdb = int(elem.attrib["dbResNum"])
                except:
                    resnum_pdb = None
                    continue
            # 'dbSource': 'UniProt', 'dbCoordSys': 'UniProt'})['dbResNum'])
            elif elem.attrib["dbSource"] == elem.attrib["dbCoordSys"] == "UniProt":
                try:
                    resnum_uniprot = int(elem.attrib["dbResNum"])
                    uniprot_id = elem.attrib["dbAccessionId"]
                except:
                    resnum_uniprot = None
                    uniprot_id = None
                    continue

    return pdb_uniprot_mapping_list_by_chain
