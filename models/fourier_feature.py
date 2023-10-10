import torch
import numpy as np

class FourierFeature(torch.nn.Module):
    def __init__(self, in_channels=1042):
        super().__init__()



        self.head = torch.nn.Sequential(torch.nn.Linear(in_features=in_channels, out_features=in_channels//2, bias=True), torch.nn.ReLU())

        self.B = torch.randn(in_channels//4, in_channels//2)
        self.B_T = self.B.T

        self.tail = torch.nn.Sequential(torch.nn.Linear(in_features=in_channels//2, out_features=1, bias=True))
                                        # torch.nn.ReLU(),
                                        #torch.nn.Linear(in_features=256, out_features=128, bias=True),
                                        # torch.nn.ReLU(),
                                        # torch.nn.Linear(in_features=128, out_features=1, bias=True))


    def forward(self, x):

        x = self.head(x)

        x_pre_fourier = x

        x = (2 * np.pi * x) @ self.B_T

        x = torch.cat(tensors=(torch.sin(x), torch.cos(x)), axis=-1)

        x = torch.squeeze(self.tail(x), dim=-1)

        x = torch.nn.functional.sigmoid(x)

        return x