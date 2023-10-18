from typing import Optional

import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, *args, **kwargs):
        super().__init__()


        self.in_channels = in_channels

        self.model = nn.LSTM(
            input_size=in_channels,
            hidden_size=in_channels//2,
            num_layers=3,
            bias=True,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.lin = nn.Linear(in_features=in_channels, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        output, (h_n, c_n) = self.model(x)

        return self.sig(self.lin(output))


class Transformer(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 1, num_heads: int = 8):
        super().__init__()

        self.input_dim, self.hidden_dim, self.output_dim, self.num_heads = input_dim, hidden_dim, output_dim, num_heads



        # 3. embedding tensor to encoding
        # self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=2048, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6)

        self.classifier = nn.Sequential(nn.Linear(in_features=input_dim, out_features=output_dim),
                                        nn.Sigmoid())

        self.start_query = nn.Embedding(1, input_dim)
        self.one_embedding = nn.Embedding(1, input_dim)
        self.zero_embedding = nn.Embedding(1, input_dim)

    def forward(self, x, label = None):
        # TODO Parallel processing

        if self.training:
            current_decoder_input = [self.get_output(x=x[i], label=label) for i in range(x.shape[0])]
        else:(
            current_decoder_input) = [self.get_output_recurrent_mode(x=x[i], starting_query=self.start_query.weight, label=label) for i in range(x.shape[0])]

        current_decoder_input = torch.nn.utils.rnn.pad_sequence(current_decoder_input, batch_first=True)
        return current_decoder_input

    def get_output(self, x:torch.Tensor, label):
        tgt_mask = torch.ones((x.shape[0] + 1, x.shape[0] + 1), dtype=x.dtype, device=x.device).tril()
        tgt = self.get_teacher_forcing_input(label)

        res = self.decoder(memory=x, tgt=tgt)[1:]
        assert not torch.isnan(res).any()
        res = self.classifier(res)
        return res

    def get_teacher_forcing_input(self, label):
        return torch.cat(
            [self.start_query.weight] + [self.one_embedding.weight if l > 0.5 else self.zero_embedding.weight for l in
                                         label.squeeze()])

    def get_output_recurrent_mode(self, x: torch.Tensor, starting_query: torch.Tensor, label: Optional[torch.Tensor] = None):
        current_decoder_input = torch.zeros(size=(x.shape[0] + 1, self.input_dim), dtype=x.dtype, device=x.device)
        current_decoder_input_teacher_forcing = current_decoder_input.clone()
        query_mask = torch.ones((x.shape[0] + 1, x.shape[0]), dtype=x.dtype, device=x.device).tril()

        current_decoder_input[0] = current_decoder_input_teacher_forcing[0] = self.start_query.weight

        if label is not None:
            label = label.squeeze()
            current_decoder_input_teacher_forcing = self.get_teacher_forcing_input(label)

        for i in range(x.shape[0]):
            query_mask[i] = 1.0

            current_input = current_decoder_input_teacher_forcing[:1+i]
            current_input_mask = query_mask[:1+i, :1+i]
            current_output = self.decoder(memory=x, tgt=current_input, tgt_mask=current_input_mask)
            current_output = current_output[-1]
            current_output = self.classifier(current_output)

            current_decoder_input[i + 1] = current_output

            if label is None:
                current_decoder_input_teacher_forcing[i + 1] = self.one_embedding if current_output.round().item() > 0.5 else self.zero_embedding



        return current_decoder_input[1:]