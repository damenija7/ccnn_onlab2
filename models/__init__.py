import torch

from models.lstm import LSTM
from models.test import TestConv3, Test3


class Mixed(torch.nn.Module):
    def __init__(self, in_channel=1024, out_channel=1):
        super().__init__()

        self.model_conv = TestConv3(in_channel, out_channel)
        self.model_lstm = LSTM(in_channel, out_channel)
        self.model_lin = Test3(in_channel, out_channel)

    def forward(self, x):
        return (self.model_conv(x) + self.model_lstm(x) + self.model_lin(x)) / 3.0



