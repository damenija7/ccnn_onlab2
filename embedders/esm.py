import torch


class ESM2Embedder:
    def __init__(self):
        #self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
        self.model.eval()
        self.device = 'cpu'

    def __call__(self, sequence):
        return self.embed(sequence)

    def embed(self, sequences):
        batch_converter = self.alphabet.get_batch_converter()

        if not isinstance(sequences, list):
            sequences = [('0', sequences)]
        else:
            sequences = [(i, sequence) for i, sequence in enumerate(sequences)]


        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        with torch.no_grad():
            batch_tokens = batch_tokens.to(self.device)
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        results = []

        for i, token_len in enumerate(batch_lens):
            results.append(token_representations[i, 1 : token_len - 1])

        if len(results) == 1:
            return results[0]

        return results


    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self