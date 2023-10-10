import random
import string

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        ran_len = random.randint(100, 600)

        labels = torch.Tensor([random.randint(0, 1) for _ in range(ran_len)])
        # sequence = ''.join(random.choice(string.ascii_uppercase) for i in range(ran_len))
        embedding = torch.rand(ran_len, 1024)

        return embedding, labels
