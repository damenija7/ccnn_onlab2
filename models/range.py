from pprint import pprint
from typing import Callable, Optional

import torch
from torch import nn

from model_training.loss import DetrLoss


class ConvRange(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()

        in_features = input_dim

        get_cnn_block = lambda in_channels, out_channel :nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channel, kernel_size=3, padding=3 // 2, padding_mode='circular'),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU())

        self.conv1 = get_cnn_block(in_features, in_features//2)
        self.conv2 = get_cnn_block(in_features//2, in_features//2)

        self.conv3 = nn.Sequential(get_cnn_block(in_features//2, in_features//4),
                                   nn.AvgPool1d(kernel_size=4),
                                   get_cnn_block(in_features//4, 3))


        self.classifier = torch.nn.Sigmoid()

    def forward(self, x, labels = None):
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
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 1, num_heads: int = 2, num_layers: int = 2,
                 sequence_embedder: Callable = None):
        super().__init__()

        if (input_dim % num_heads) != 0:
            self.preprocess = nn.Linear(input_dim, 64*num_heads)
            input_dim = 64*num_heads
        else:
            self.preprocess = lambda x : x

        self.input_dim, self.hidden_dim, self.output_dim, self.num_heads = input_dim, hidden_dim, output_dim, num_heads

        self.loss_func = DetrLoss(0.9)

#        if not sequence_embedder:
#            raise Exception("Must specify a sequence embedder")


        # 3. embedding tensor to encoding
        # self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=3)

        self.class_embed = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.box_embed = nn.Sequential(nn.Linear(input_dim, input_dim),
                                       nn.ReLU(),
                                       nn.Linear(input_dim, 2),
                                       nn.Sigmoid())

        self.output_query =  nn.Embedding(20, input_dim)

    def forward(self, x, label: Optional = None):
        if isinstance(x, list):
            x = [self.preprocess(x_i) for x_i in x]
        else:
            x = self.preprocess(x)

        # TODO Parallel processing
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        x_padded: torch.Tensor = pad_sequence(x, batch_first=True, padding_value=-float('inf'))
        # (batch_siz, seq)
        # if True, then masked
        padding_mask_bool = ~(x_padded[:, :, 0].squeeze(dim=-1) > -float('inf'))
        padding_mask_float = padding_mask_bool.to(x_padded.dtype)
        padding_mask_float[~padding_mask_bool] = 0.0
        padding_mask_float[padding_mask_bool] = -float('inf')

        x_padded = torch.nan_to_num(x_padded, neginf=0.0)

        memory_mask = torch.cat([padding_mask_float.unsqueeze(dim=-2)] * self.output_query.weight.shape[0], dim=1)




        tgt = self.output_query.weight.expand(x_padded.shape[0], *self.output_query.weight.shape)
        assert not torch.any(torch.isnan(self.output_query.weight))

        # account for multiheadedness
        memory_mask = torch.cat([memory_mask] * self.num_heads)
        tgt_mask = torch.zeros(dtype=memory_mask.dtype, device=tgt.device, size=(tgt.shape[0] * 8, tgt.shape[1], tgt.shape[1]))
        #for i in range(self.output_query.weight.shape[0]):
        #    tgt_mask[:, i, i] = -float('inf')

        output = self.decoder(memory=x_padded, tgt=tgt,
                              memory_key_padding_mask=padding_mask_float)
        assert not torch.any(torch.isnan(output))
        output = torch.cat([self.box_embed(output), self.class_embed(output)], dim=-1)

        if label is None:
            return output

        # TEST
        # output = self.get_label_output(label, output)

        loss = self.loss_func(output, label)


        return output, loss

    def get_label_output(self, label, output):
        # TODO FOR TESTING PURPOSES
        label_output = torch.zeros(size=(len(label), max(len(label_i) for label_i in label), output.shape[-1]),
                                   dtype=output.dtype, device=output.device)
        for i, label_i in enumerate(label):
            label_output[i, :len(label_i), :-1] = label_i
            label_output[i, :len(label_i), -1] = 1.0
        return label_output

    @property
    def statistics(self):
        query_similarity = torch.var(self.output_query.weight, dim=0)

        stat = {
            "query_similarity": (query_similarity.min().item(), query_similarity.max().item()),
            # "classifier_similarity": (classifier_similarity.min().item(), classifier_similarity.max().item())
        }
        return stat




