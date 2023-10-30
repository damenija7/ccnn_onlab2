from torch import nn


class ConvBasic(nn.Module):
    def __init__(self, num_dim: int = 1024):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(15, 1), padding=(15//2, 0)),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(15, 1), padding=(15//2, 0)),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)

        x = self.model(x)
        x = self.classifier(x)
        x = x.squeeze(dim=-1).permute(0,2,1)

        return x
