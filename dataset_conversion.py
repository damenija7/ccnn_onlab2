import hashlib
import logging
import os
import pickle
import re
import numpy as np
import csv
from pprint import pprint
from typing import Dict, Set, List, Tuple

from Bio import SeqIO
from tqdm.auto import tqdm

from dataset_conversion import helper
from dataset_conversion.helper import get_online_resource, get_uniprot_sequence, get_pdb_uniprot_mapping_by_chain


def get_uniprot_dict(csv_path: str, cache_folder_path: str):
    """
    @return: Dict: Uniprot_ID -> PDB_ID -> Chain -> Set of (CC Region Range Begin, CC Region Rage End) entries.
    """
    # Acc/uniprot_id -> PDB_ID -> CHAIN) -> [CC_REGION_RES_IDX]
    # Indexing with the uniprot id then its corresponding pdb id then the chain name gives the set of residue indices that belong to a coiled coil region
    # ( Return value )
    uniprot_to_pdb_id_chain_dict: Dict[str, Dict[str, Dict[str, Set[int]]]] = {}

    # get number of lines so that we can show the progress bar
    with open(csv_path, "r") as csv_file:
        num_lines: int = -15 + len(csv_file.readlines())


    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # Skip first 15 lines ( they dont contain data )
        for _ in range(15):
            next(csv_reader)

        for ccp_entry in tqdm(csv_reader, total=num_lines, desc="Converting CSV Entries to Uniprot Data"):
            pdb_id: str = ccp_entry[1]
            chains: Set[str] = set(
                [
                    helix_range_dat.split(":")[1]
                    # Seventh column contains coiled coil range data (see: input csv file)
                    for helix_range_dat in ccp_entry[7].split(" ")
                ]
            )

            # For given pdb id lists CC region corresponding to indexed chain
            helix_ranges_by_chain: Dict[str, List[Tuple[int, int]]] = {
                chain: [] for chain in chains
            }

            helix_ranges = ccp_entry[7].split(" ")
            for helix_range in helix_ranges:
                h_range_beg, h_range_end, chain = re.match(
                    r"([-\d]+)-([-\d]+):([\d\w]+)", helix_range
                ).groups()
                h_range = (int(h_range_beg), int(h_range_end))
                helix_ranges_by_chain[chain].append(h_range)

            # If biological assembly, solve chain naming
            # TODO Not done biological assembly
            if "_ba" in pdb_id:
                # TODO Ignore biological assembly entries for now
                continue

            try:
                #
                # SIFTS contains  residue-level mapping between UniProt and PDB entries.
                #
                sifts_download_url: str = (
                    f"ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id}.xml.gz"
                )
                sifts_access_path: str = helper.get_online_resource(
                    download_url=sifts_download_url,
                    cache_folder_path = cache_folder_path
                )
                # chain -> pdb_res_idx -> (uniprot id, uniprot_res_idx) dict for given pdb entry
                pdb_uniprot_mapping_by_chain = helper.get_pdb_uniprot_mapping_by_chain(
                    sifts_access_path
                )

                for chain in helix_ranges_by_chain:
                    # If chain has no corresponding sifts mapping then ignore
                    if chain not in pdb_uniprot_mapping_by_chain:
                        continue

                    for helix_range_beg, helix_range_end in helix_ranges_by_chain[
                        chain
                    ]:
                        for pdb_res_idx in range(helix_range_beg, helix_range_end + 1):
                            try:
                                # NOTE: Each Residue can have/has its own separate uniprot id entry associated with it
                                (
                                    uniprot_id,
                                    uniprot_res_idx,
                                ) = pdb_uniprot_mapping_by_chain[chain][pdb_res_idx]
                                # Add mapped uniprot residue idx to uniprot_id->pdb_id->chain associated Set
                                uniprot_to_pdb_id_chain_dict.setdefault(
                                    uniprot_id, {}
                                ).setdefault(pdb_id, {}).setdefault(chain, set()).add(
                                    uniprot_res_idx
                                )
                            # Log
                            except Exception as e:
                                e_txt = f"{pdb_id}:{chain}:{pdb_res_idx} -> No SIFTS Mapping"
                                logging.info(
                                  e_txt
                                )
                                print(e_txt)
            except Exception as e:
                logging.info(e)
                print(e)

    return uniprot_to_pdb_id_chain_dict


"""
Convert Dataset to Pytorch readable format

Aka: (uniprot sequence, is coiled coil region bitmask )
"""
def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    # Cache stores data like the sifts database, uniprot entry Protein Sequences etc..
    parser.add_argument("-c", "--cache", default="cache/")

    args = parser.parse_args()

    pprint(vars(args))

    csv_input_path: str = args.input
    csv_output_path: str = args.output if args.output else args.input + ".out.csv"

    slice_idx = csv_output_path.rindex(".csv")

    csv_output_path_by_label = csv_output_path[:slice_idx] + ".bylabel.csv"
    cache_folder_path = args.cache
    
    # If Input already processed into Intermediate uniprot_data, then load it from cache !
    #
    BUF_SIZE = 32768 # Read in 32kb chunks
    sha1 = hashlib.sha1()
    with open(csv_input_path, 'rb') as f:
        dat = f.read(BUF_SIZE)
    if not dat:
                sha1.update(dat)
    
    sha_digest = sha1.hexdigest()
    logging.info(f"Input CSV SHA1 Digest: {sha_digest}")
    uniprot_data_cache_path = cache_folder_path + "input_" + sha_digest + ".pkl"

    #
    if os.path.isfile(uniprot_data_cache_path):
        logging.info("Loading Already Processed Uniprot Data of Input...")
        with open(uniprot_data_cache_path, 'rb') as f:
            uniprot_data = pickle.load(f)

    # If Input already processed into Intermediate uniprot_data, then load it from cache !
    else:
        logging.info("Processing Input into Uniprot Data...")
        uniprot_data = get_uniprot_dict(csv_input_path, cache_folder_path)

        with open(uniprot_data_cache_path, 'wb') as f:
            pickle.dump(uniprot_data, f)

    import csv


    # Load UniProtKB/Swiss-Prot Database Sequence Data Into 'uniprot_sequences' Dict
    #
    #
    uniprot_sprot_download_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    uniprot_sprot_access_path = get_online_resource(
        download_url=uniprot_sprot_download_url,
        resource_loader_func=lambda x: x,
        cache_folder_path=cache_folder_path,
    )
    uniprot_sequences: Dict[str, str] = {
        seq.name.split("|")[1]: str(seq.seq)
        for seq in SeqIO.parse(uniprot_sprot_access_path, "fasta")
    }



    # Write OUTPUT in (Uniprot Id, Residue Idx, Amino Acid Value) format
    # for every Residue belonging to a coiled coil region !
    with open(csv_output_path, "w") as f_output_csv:
        csv_writer = csv.writer(f_output_csv)
        csv_writer.writerow(("acc", "resid_idx", "aa_val"))


        uniprot_id: str
        for uniprot_id in tqdm(uniprot_data, desc="Writing Output"):
            try:
                uniprot_seq: str = get_uniprot_sequence(uniprot_id, cache_folder_path, uniprot_sequences)
            except Exception as e:
                print(e)
                logging.warning(e)

            helix_range_by_chain_by_pdb_ids = uniprot_data[uniprot_id]

            for pdb_id in helix_range_by_chain_by_pdb_ids:
                helix_range_by_chain = helix_range_by_chain_by_pdb_ids[pdb_id]
                for chain in helix_range_by_chain:
                    for uniprot_res_idx in helix_range_by_chain[chain]:
                        try:
                            aa_val = uniprot_seq[uniprot_res_idx - 1]
                            csv_writer.writerow((uniprot_id, uniprot_res_idx, aa_val))
                        except Exception as e:
                            print("Failed during {pdb_id}:{chain} -> UNIPROT:{uniprot_id}:res{uniprot_res_idx} ({e})")
                            logging.info(e)




    uniprot_seq_dict = {}
    uniprot_label_dict = {}


    # Create Uniprot Coiled Coil Region Bitmasks based on just created output !
    with open(csv_output_path, "r") as f_output_csv:
        csv_reader = csv.reader(f_output_csv)
        next(csv_reader)

        for row in csv_reader:
            uniprot_id = row[0]
            if uniprot_id not in uniprot_seq_dict:
                uniprot_seq_dict[uniprot_id] = get_uniprot_sequence(uniprot_id, cache_folder_path, uniprot_sequences)
                uniprot_label_dict[uniprot_id] = np.zeros(
                    len(uniprot_seq_dict[uniprot_id]), dtype=np.bool_
                )

            resid_idx = int(row[1]) - 1
            uniprot_label_dict[uniprot_id][resid_idx] = True

    # Write OUTPUT in (Uniprot Id, Uniprot Sequence, Uniprot Coiled Coil Region Bitmask) format
    with open(csv_output_path_by_label, "w") as f_output_csv:
        csv_writer = csv.writer(f_output_csv)
        csv_writer.writerow(("uniprot_id", "sequence", "label"))
        for uniprot_id in uniprot_seq_dict:
            uniprot_label = "".join(
                str(int(bool_val)) for bool_val in uniprot_label_dict[uniprot_id]
            )
            uniprot_seq = uniprot_seq_dict[uniprot_id]

            assert len(uniprot_seq) == len(uniprot_label)

            csv_writer.writerow((uniprot_id, uniprot_seq, uniprot_label))


if __name__ == "__main__":
    main()
