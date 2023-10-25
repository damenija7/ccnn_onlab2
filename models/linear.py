from torch import nn


class Residual(nn.Module):
    def __init__(self, block_1, block_2):
        super().__init__()
        self.block_1, self.block_2 = block_1, block_2

    def forward(self, x):
        return self.block_2(x + self.block_1(x))

class Linear(nn.Module):
    def __init__(self, input_channels: int = 1024, num_residual_blocks: int = 2):
        super().__init__()

        self.input_channels = 1024

        def get_lin_block(in_channel, out_channel, norm = True, activation=nn.ReLU, dropout=False):
            modules = [
                nn.Linear(in_features=in_channel, out_features=out_channel)
            ]
            if norm:
                modules.append(nn.LayerNorm(normalized_shape=out_channel))
            if activation is not None:
                modules.append(activation())
            if dropout:
                modules.append(nn.Dropout())

            return nn.Sequential(*modules)

        def get_residual_con_block(in_channel, out_channel, activation=nn.ReLU, dropout=False):


            block_1 = get_lin_block(in_channel, in_channel, False, None, dropout)
            block_2 = get_lin_block(in_channel, out_channel, True, activation, dropout)

            return Residual(block_1, block_2)

        #self.l1 = get_lin_block(input_channels, input_channels)
        #self.l2 = get_lin_block(input_channels, input_channels//2)
        #self.l3 = get_lin_block(input_channels//2, input_channels//2)
        #self.l4 = get_lin_block(input_channels//2, input_channels//4)
        #self.l5 = get_lin_block(input_channels//4, input_channels//4)
        #self.l6 = get_lin_block(input_channels//4, 1, False, activation=nn.Sigmoid)
        model_layers = []

        for i in range(num_residual_blocks):
            model_layers.append(get_residual_con_block(input_channels, input_channels//2, dropout=True))
            input_channels //= 2
        model_layers.append(get_lin_block(input_channels, 1, norm=False, activation=nn.Sigmoid))

        self.model = nn.Sequential(*model_layers)


    def forward(self, x, labels = None):
        return self.model(x)