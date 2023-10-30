from torch._C.cpp import nn


class GAN(nn.Module):
    def __init__(self):
        super().__init__()