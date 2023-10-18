import math
from typing import Callable

import numpy as np
import torch
import torchvision
from torch import nn
from torch import functional as F


class ConvRange(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()

        in_features = input_dim

        get_cnn_block = lambda in_channels, out_channel :nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channel, kernel_size=3, padding=3 // 2),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU())

        self.conv1 = get_cnn_block(in_features, in_features//2)
        self.conv2 = get_cnn_block(in_features//2, in_features//2)

        self.conv3 = nn.Sequential(get_cnn_block(in_features//2, in_features//4),
                                   nn.AvgPool1d(kernel_size=4),
                                   get_cnn_block(in_features//4, 3))


        self.classifier = torch.nn.Sigmoid()

    def forward(self, x):
        x_resized = torch.stack([
            torch.nn.functional.interpolate(x[i].permute(1, 0).unsqueeze(-2), size=[1024]).squeeze(dim=-2).permute(1, 0)
            for i in range(len(x))], dim=0)

        x_output_orig = x_resized.permute(0, -1, -2)
        x_output = x_output_orig

        x_output = self.conv1(x_output)
        x_output = x_output + self.conv2(x_output)
        x_output = self.conv3(x_output)

        x_output = x_output.permute(0, -1, -2)

        x_output = self.classifier(x_output)

        return self.classifier(x_output)




class Transformer(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 1, num_heads: int = 8,
                 sequence_embedder: Callable = None):
        super().__init__()

        self.input_dim, self.hidden_dim, self.output_dim, self.num_heads = input_dim, hidden_dim, output_dim, num_heads

#        if not sequence_embedder:
#            raise Exception("Must specify a sequence embedder")


        # 3. embedding tensor to encoding
        # self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=4096, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6)

        self.classifier = nn.Sequential(nn.Linear(in_features=input_dim, out_features=3),
                                        nn.Sigmoid())

        self.output_query =  nn.Embedding(20, input_dim)

    def forward(self, x, padding_mask = None):
        # TODO Parallel processing
        # x = [torch.nn.functional.interpolate(x[i].permute(1, 0).unsqueeze(-2), size=[1024]).squeeze(dim=-2).permute(1, 0) for i in range(len(x))]

        output = torch.stack([self.decoder(memory=x[i], tgt=self.output_query.weight) for i in range(len(x))])
        output = self.classifier(output)


        return output


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





