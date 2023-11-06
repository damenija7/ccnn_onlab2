import torch


import esm
from openfold.utils.feats import atom14_to_atom37
from openfold.np import residue_constants

class ESMFoldEmbedder:
    def __init__(self):
        #self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.model = esm.pretrained.esmfold_v1()
        #self.model = torch.hub.load("facebookresearch/esm:main", "esmfold_v1")
        self.model.eval()
        self.device = 'cpu'

        self.atom_types = residue_constants.atom_types
        self.ca_pos = self.atom_types.index('CA')

    def __call__(self, sequence):
        return self.embed(sequence)

    def embed(self, sequences):
        if not isinstance(sequences, list):
            sequences = [sequences]

        with torch.no_grad():
            results = [self.to_output(self.model.infer(seq)) for seq in sequences]


        # self.model.infer_pdb(sequences[0])

        if len(results) == 1:
            return results[0]

        return results

    def to_output(self, output: dict):
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output).squeeze(dim=0)
        final_atom_positions = final_atom_positions[:, self.ca_pos]
        # final_atom_positions = final_atom_positions.cpu().numpy()

        return final_atom_positions


    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self