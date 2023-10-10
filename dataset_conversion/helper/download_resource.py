import gzip
import logging
import os
import shutil
import urllib
from typing import Callable, Dict
import json
import requests
from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaIterator


def get_online_resource(
        download_url: str,
        resource_loader_func: Callable = lambda resource_path: resource_path,
        cache_folder_path: str = "./cache/",
):
    """
    Downloads a resource (or obtains it from cache if already downloaded) and returns it in a form specified by resource_loader_func

    NOTE:
        Make sure every file name is unique (returns first downloaded file otherwise) !

    :param download_url: File Online Url
    :param resource_loader_func: Function that uses file path to load/process the file.
    :param cache_folder_path: Folder to cache all the resources
    :return: File processed with resource_loader_func ( returns file path by default )
    """

    resource_name: str = download_url.split("/")[-1]
    # If resource is compressed initially save uncompressed one
    if resource_name.endswith(".gz"):
        resource_name = resource_name[:-3]
    resource_path: str = cache_folder_path + resource_name

    # download resource if it doesnt exist
    if not os.path.exists(resource_path):
        if not os.path.exists(cache_folder_path):
            os.makedirs(cache_folder_path)

        logging.info(f'Downloading "{download_url}"')
        urllib.request.urlretrieve(
            download_url,
            resource_path,
        )
        # Uncompress if compressed and delete leftover compressed archive
        if download_url.endswith(".gz"):
            gz_path: str = resource_path
            resource_path = gz_path[:gz_path.rindex(".gz")]
            with gzip.open(gz_path, "rb") as f_in:
                with open(resource_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)

    # Process Resource and Return ( Just return path on default )
    return resource_loader_func(resource_path)


def get_uniprot_fasta(uniprot_id: str, cache_folder_path: str = "./cache/") -> FastaIterator:
    fasta_download_url = f"http://www.uniprot.org/uniprot/{uniprot_id}.fasta"

    def load_fasta_func(resource_path: str) -> str:
        return SeqIO.parse(resource_path, "fasta")

    return get_online_resource(
        download_url=fasta_download_url,
        resource_loader_func=load_fasta_func,
        cache_folder_path=cache_folder_path,
    )




def get_uniprot_sequence(uniprot_id: str, cache_folder_path, uniprot_sequences: Dict[str, str]) -> str:
    """
    Searches uniprot sequence in  (UniProtKB/Swiss-Prot ->  UniProtKB Non Curated Part -> UniProtKB Archived Part) Order !
    """
    if uniprot_id in uniprot_sequences:
        return uniprot_sequences[uniprot_id]

    # Uniprot Entry not in curated UniProtKB/Swiss-Prot database,
    # -> look for entry outside of it in non curated part
    uniprot_seq_path: str = (
            cache_folder_path + uniprot_id + ".txt"
    )

    if not os.path.exists(uniprot_seq_path):
        response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}")
        response_json = json.loads(response.content)
        # get sequence
        try:
            sequence = response_json["sequence"]["value"]
        except Exception:
            #
            # Exception: Entry is Inactive/Outdated: Need to find latest archived version
            #
            if response_json.get("entryType", "") == "Inactive":
                try:
                    response_entry_versions = json.loads(
                        requests.get(
                            f"https://rest.uniprot.org/unisave/{uniprot_id}?format=json"
                        ).content
                    )["results"]

                    max_version = max(
                        int(entry["entryVersion"]) for entry in response_entry_versions
                    )
                    response_entry = requests.get(
                        f"https://rest.uniprot.org/unisave/{uniprot_id}?format=txt&versions={max_version}"
                    )
                    lines = response_entry.text.split("\n")

                    start_idx, end_idx = None, None
                    for line_idx, line in enumerate(lines):
                        if line.startswith("SQ"):
                            start_idx = line_idx + 1
                        if line.startswith("//"):
                            end_idx = line_idx

                    sequence = "".join(lines[start_idx:end_idx]).replace(" ", "")
                except:
                    raise Exception(response.text)
            else:
                raise Exception(response.text)
        # write sequence
        with open(uniprot_seq_path, "w") as f:
            f.write(sequence)
        return sequence

    with open(uniprot_seq_path, "r") as f:
        return f.readlines()[0]
