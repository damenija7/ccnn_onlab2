from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, *args, **kwargs):
        super().__init__()


        self.in_channels = in_channels

        self.model = nn.LSTM(
            input_size=in_channels,
            hidden_size=in_channels//2,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.lin = nn.Linear(in_features=in_channels, out_features=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        output, (h_n, c_n) = self.model(x)

        return self.sig(self.lin(output))