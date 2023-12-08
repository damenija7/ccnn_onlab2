import logging
import os.path
import sys
from typing import Callable, List, Optional

import h5py
import numpy as np
import torch
from torch import Tensor


class EmbeddingCacher:
    def __init__(self, sequence_embedder: Callable, cache_path: str):
        self.embedder = sequence_embedder
        self.cache_path = cache_path


        self.stored_secs = None

        if not os.path.exists(cache_path):
            f = h5py.File(cache_path, "x")
            f.close()

    def cache_embeddings(
            self, sequences: List[str]
    ):
        sequences_to_cache = sequences
        try:
            # Filter sequences so that only uncached ones are embedded and saved (save performance)
            with h5py.File(self.cache_path, "r") as cache_file:
                cache_keys = cache_file.keys()
                sequences_to_cache = [sequence for sequence in sequences if sequence not in cache_keys]
        except:
                pass

        # Return if input empty or else exception would occur in embedder
        if len(sequences_to_cache) == 0:
            return

        embeddings_to_cache = [self.embedder(sequence) for sequence in sequences_to_cache]
        self._save_local_cache(sequences_to_cache, embeddings_to_cache)

    def _save_local_cache(self, sequences: List[str], embeddings: List[str]):
        try:
            with h5py.File(self.cache_path, "a") as cache_file:
                for sequence, embedding in zip(sequences, embeddings):
                    if sequence not in cache_file.keys():
                        cache_file.create_dataset(
                            name=sequence,
                            data=np.array(embedding.cpu()),
                            chunks=True,
                            compression="gzip",
                        )

        except Exception as e:
            error_message = f"Tried saving Embedding Cache {self.cache_path} but failed"
            print(error_message)
            logging.warning(error_message)
            print(f"\t {e}")
            logging.warning(f"\t {e}")

    def embed(self, sequence: str):
        return self.get_embeddings([sequence])[0]

    def get_embeddings(self, sequences: List[str]) -> List[Tensor]:
        embeddings: List[Tensor] = []

        with h5py.File(self.cache_path, "r") as cache_file:
            if self.stored_secs is None:
                self.stored_secs = set(cache_file.keys())

            sequences_not_cached = [seq for seq in sequences if seq not in cache_file.keys()]
            if len(sequences_not_cached) > 0:
                    cache_file.close()
                    self.cache_embeddings(sequences_not_cached)
                    self.stored_secs.update(sequences_not_cached)
                    cache_file = h5py.File(self.cache_path, "r")

            embeddings = [torch.Tensor(cache_file[seq][:]) for seq in sequences]

        return embeddings

    def __call__(self, sequences: List[str]) -> List[Tensor]:
        if isinstance(sequences, list):
            return self.get_embeddings(sequences)

        return self.get_embeddings([sequences])[0]