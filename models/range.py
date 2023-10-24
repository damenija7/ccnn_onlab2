from typing import Callable

import torch
from torch import nn


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
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        x_padded: torch.Tensor = pad_sequence(x, batch_first=True, padding_value=-float('inf'))
        # (batch_siz, seq)
        padding_mask = (x_padded[:, :, 0].squeeze(dim=-1) > -float('inf')).float()
        x_padded = torch.nan_to_num(x_padded, neginf=0.0)

        memory_mask = torch.cat([padding_mask.unsqueeze(dim=-2)] * self.output_query.weight.shape[0], dim=1)

        tgt = self.output_query.weight.expand(x_padded.shape[0], *self.output_query.weight.shape)

        # account for multiheadedness
        memory_mask = torch.cat([memory_mask] * 8)

        output = self.decoder(memory=x_padded, memory_mask=memory_mask, tgt=tgt)
        output = self.classifier(output)

        return output





