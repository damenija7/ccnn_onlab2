from torch import nn

from models.linear import Linear
from models.rnn import LSTM


class EnsembleLinearLSTM(nn.Module):
    def __init__(self, input_channels, weights = [0.5, 0.5]):
        super().__init__()
        self.model_1 = Linear(input_channels)
        self.model_2 = LSTM(input_channels)
        self.weights = weights

    def forward(self, x):
        return self.weights[0] * self.model_1(x) + self.weights[1] * self.model_2(x)