from typing import Optional, List

import torch
from torch import nn

from models import PositionalEncoding
from torch.nn.utils.rnn import pad_sequence


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

class EmbeddingModel(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.pos_encoder = nn.Embedding(5000, input_dim)
        self.one_embedding = nn.Embedding(1, input_dim)
        self.zero_embedding = nn.Embedding(1, input_dim)

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=input_dim, out_features=input_dim)
        )

    def forward(self, x: List[torch.Tensor], label: Optional[List[torch.Tensor]]):
        model_output = self.model(x)

        preds = self.get_preds(model_output)

        if label is not None:
            loss = self.get_loss(label, model_output)
            return preds, loss
        return preds

    def get_preds(self, model_output):
        sim_one = self.get_similarity(model_output, self.model(self.get_label_embedding_repr(model_output, 1)))
        sim_zero = self.get_similarity(model_output, self.model(self.get_label_embedding_repr(model_output, 0)))
        res = torch.zeros(size=(model_output.shape[0], model_output.shape[1], 1), dtype=model_output.dtype, device=model_output.device)

        pos_mask = sim_one > sim_zero
        res[pos_mask] = 1.0

        return res

    def get_loss(self, label, model_output):
        #res = self.get_similarity(model_output, self.get_label_embedding_repr(label))
        label_embedding_output = self.model(self.get_label_embedding_repr(label))



        similarity = self.get_similarity(model_output, label_embedding_output)


        loss = nn.MSELoss(reduction='none')(model_output, label_embedding_output).mean(dim=-1)
        loss = loss + nn.MSELoss(reduction='none')(similarity, torch.ones_like(similarity))

        weights = torch.zeros_like(loss)
        weights[label > 0.5] = 0.9
        weights[label <= 0.5] = 0.1


        return loss * weights

    def get_label_embedding_repr(self, label, force_label: Optional[int] = None):
        res = torch.zeros(dtype=label.dtype, device=label.device, size=(label.shape[0], label.shape[1], self.zero_embedding.weight.shape[-1]))
        label = label.squeeze(dim=-1)

        if force_label is None:
            pos_mask = label > 0.5
            res[pos_mask] = self.one_embedding.weight[0]
            res[~pos_mask] = self.zero_embedding.weight[0]
        else:
            if force_label > 0.5:
                res[:] = self.one_embedding.weight[0]
            else:
                res[:] = self.zero_embedding.weight[0]

        for i in range(label.shape[0]):
            res[i] = self.get_pos_embedded_version(res[i])
            res[i] = self.get_pos_embedded_version(res[i])

        return res

    def get_similarity_single_embedding(self, model_output, embedding) -> torch.Tensor:
        res = (model_output * embedding[None, None, ::]).sum(axis=-1) / torch.norm(model_output, dim=-1) / torch.norm(embedding, dim=-1)
        #res = (model_output * embedding[None, None, ::]).sum(axis=-1)
        #res = (model_output - embedding[None, None, ::]).pow(2).sum(axis=-1)
        return res

    def  get_similarity(self, pred, label):
        res = (pred * label).sum(axis=-1) / torch.norm(pred, dim=-1) / torch.norm(label, dim=-1)
        #res = (pred * label).sum(axis=-1)
        # res = (pred - label).pow(2).sum(axis=-1)
        return res

    def get_pos_embedded_version(self, x):
        return x + self.pos_encoder.weight[:len(x)]


class Transformer(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 1, num_heads: int = 8):
        super().__init__()

        self.input_dim, self.hidden_dim, self.output_dim, self.num_heads = input_dim, hidden_dim, output_dim, num_heads

        self.pos_encoder = PositionalEncoding(input_dim)
        self.pos_encoder_alt = nn.Embedding(5000, input_dim)

        # 3. embedding tensor to encoding
        # self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=2048, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6)

        self.classifier = nn.Sequential(nn.Linear(in_features=input_dim, out_features=output_dim),
                                        nn.Sigmoid())

        self.start_query = nn.Embedding(1, input_dim)
        self.one_embedding = nn.Embedding(1, input_dim)
        self.zero_embedding = nn.Embedding(1, input_dim)

    def forward(self, x: List[torch.Tensor], label: Optional[List[torch.Tensor]] = None):
        # TODO Parallel processing

        if self.training:
            x_padded: torch.Tensor = pad_sequence(x, batch_first=True, padding_value=-float('inf'))
            padding_mask = (x_padded[:, :, 0].squeeze(dim=-1) > -float('inf')).float()
            x_padded = torch.nan_to_num(x_padded, neginf=0.0)
            label_padded: torch.Tensor = pad_sequence(label, batch_first=True, padding_value=0.0)
            current_decoder_input = self.get_output(x_padded, label_padded, padding_mask)
        else:(
            current_decoder_input) = pad_sequence([self.get_output_recurrent_mode(x=x[i], starting_query=self.start_query.weight, label=None) for i in range(x.shape[0])],
                                                  batch_first=True)

        return current_decoder_input

    def get_output(self, x:torch.Tensor, label: torch.Tensor, padding_mask: torch.Tensor):
        tgt = self.get_teacher_forcing_input(label)


        mem_mask, tgt_mask = self.get_mem_tgt_mask(padding_mask)

        res = self.decoder(memory=x, memory_mask=mem_mask, tgt=tgt, tgt_mask=tgt_mask)[:, 1:, :]
        res = self.classifier(res)
        return res

    def get_mem_tgt_mask(self, padding_mask):
        # padding part in tgt_mask, mem_mask
        tgt_mask = torch.cat([torch.ones_like(padding_mask[:, 0]).unsqueeze(dim=-1), padding_mask], dim=-1)
        mem_mask = tgt_mask[:, :, None] * padding_mask[:, None, :]
        tgt_mask = tgt_mask[:, :, None] * tgt_mask[:, None, :]
        # causal mask for tgt
        # tgt_mask = tgt_mask.tril()
        # ACCOUNT FOR MULTI HEAD SOLUTION
        tgt_mask = torch.cat([tgt_mask] * 8, dim=0)
        mem_mask = torch.cat([mem_mask] * 8, dim=0)
        return mem_mask, tgt_mask

    def get_teacher_forcing_input(self, batched_label):
        return pad_sequence([torch.cat(
            [self.get_start_query()] + [self.get_embedding(l, i+1) for i, l in
                                         enumerate(label.squeeze())]) for label in batched_label], batch_first=True)

    def get_output_recurrent_mode(self, x: torch.Tensor, starting_query: torch.Tensor, label: Optional[torch.Tensor] = None):
        current_decoder_input = torch.zeros(size=(x.shape[0] + 1, self.input_dim), dtype=x.dtype, device=x.device)
        current_decoder_input_teacher_forcing = current_decoder_input.clone()
        query_mask = torch.ones((x.shape[0] + 1, x.shape[0]), dtype=x.dtype, device=x.device).tril()

        output_result = torch.zeros(size=(x.shape[0], 1), dtype=x.dtype, device=x.device)

        current_decoder_input[0] = current_decoder_input_teacher_forcing[0] = self.get_start_query()

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

            current_decoder_input[i + 1] = self.get_embedding(current_output, i+1)

            if label is None:
                current_decoder_input_teacher_forcing[i + 1] = self.get_embedding(current_output, i+1)

            output_result[i]=current_output

        return output_result



    def get_start_query(self) -> torch.Tensor:
        res= self.start_query.weight
        #res = res + self.pos_encoder.pe[0]
        res = res + self.get_pos_embedding(0)
        return res

    def get_embedding(self, x, index) -> torch.Tensor:
        res = self.one_embedding.weight if x.round().item() > 0.5 else self.zero_embedding.weight
        #res = self.start_query.weight
        #res = res + self.pos_encoder.pe[index]
        res = res + self.get_pos_embedding(index)
        return res

    def get_pos_embedding(self, index):
        return self.pos_encoder_alt.weight[index]