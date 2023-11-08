from torch import nn


class ConvBasic(nn.Module):
    def __init__(self, num_dim: int = 1024):
        super().__init__()

        kernel_siz = 31

        self.model = nn.Sequential(
            nn.Conv2d(num_dim, 64, kernel_size=(kernel_siz,1), padding=(kernel_siz//2,0), padding_mode='circular'),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2d(64, 32, kernel_size=(kernel_siz, 1), padding=(kernel_siz // 2, 0), padding_mode='circular'),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2d(32, 16, kernel_size=(kernel_siz, 1), padding=(kernel_siz // 2, 0), padding_mode='circular'),
            nn.ReLU(),
            nn.Dropout(0.25),


        )

        self.classifier = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(kernel_siz,1), padding=(kernel_siz//2,0), padding_mode='circular'),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)

        x = self.model(x)
        x = self.classifier(x)
        x = x.squeeze(dim=-1).permute(0,2,1)

        return x
