import re
from typing import Iterable, List, Union

import torch
from torch import Tensor
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    BatchEncoding,
    BertTokenizer,
    BertModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions



class ProtTransSequenceEmbedder:
    def __init__(self, transformer_link: str = "Rostlab/prot_t5_xl_half_uniref50-enc"):
        # super().__init__()
        # Change Model Class Depending on what is specified by transformer link ( Bert/ T5 / .. )
        transformer_link_lower: str = transformer_link.lower()
        if "t5" in transformer_link_lower:
            self.tokenizer = T5Tokenizer.from_pretrained(
                transformer_link, do_lower_case=False, legacy=False
            )
            self.encoder = T5EncoderModel.from_pretrained(transformer_link)
        elif "bert" in transformer_link_lower:
            self.tokenizer = BertTokenizer.from_pretrained(
                transformer_link, do_lower_case=False
            )
            self.encoder = BertModel.from_pretrained(transformer_link)


        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.encoder.to(self.device)

    def to(self, device):
        self.device = device



    def embed(
        self, sequence: str
    ) -> Union[List[Tensor], Tensor]:
        # sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        sequence_list = [sequence]
        sequence_list: List[str] = [
            " ".join(re.sub("U|Z|O", "X", sequence)) for sequence in sequence_list
        ]

        ids: BatchEncoding = self.tokenizer.batch_encode_plus(
            sequence_list, add_special_tokens=True, padding="longest"
        )
        input_ids: Tensor = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask: Tensor = torch.tensor(ids["attention_mask"]).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding: BaseModelOutputWithPastAndCrossAttentions = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # requires_grad is already False
            # embedders.last_hidden_state.requires_grad = False

        return embedding.last_hidden_state.to(torch.float32).squeeze()[:len(sequence)]

    def __call__(self, sequence: str):
        return self.embed(sequence)
