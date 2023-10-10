import os
import sys

import torch.cuda


class Config:
    transformer_link: str = "Rostlab/prot_t5_xl_half_uniref50-enc"
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    cache_folder: str = os.getcwd() + "/cache/"

    max_seq_len: int = 1000


if not os.path.exists(Config.cache_folder):
    os.makedirs(Config.cache_folder)
