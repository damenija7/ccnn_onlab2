from typing import Sequence, Iterator

import torch
from torch.utils.data import Sampler
from torch.nn.functional import gumbel_softmax

class GumbelMaxWeightedRandomSampler(Sampler[int]):
    #
    # Sample from multinomial distribution using logits
    #
    #
    # P(K=k) = argmax_{k in {1,...,length_logits}} (logit_k + z_k)
    #
    # z_k is ~ Gumbel(0,1)
    # Where Gumbel is:
    #   PDF f(x) = e(-x + -e^(-x))
    #   CDF F(x) = e^(-e^(-x))

    def __init__(self, logits: Sequence[float], num_samples: int, generator=None) -> None:
        self.logits, self.num_samples, self.generator = logits, num_samples, generator
        self.logits = torch.as_tensor(self.logits, dtype=torch.double)

    def __iter__(self) -> Iterator[int]:
        sampled_indices = [gumbel_softmax(self.logits).argmax().item() for _ in range(self.num_samples)]

        yield from iter(sampled_indices)

    def __len__(self) -> int:
        return self.num_samples