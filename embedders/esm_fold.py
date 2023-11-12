import torch


import esm
from openfold.utils.feats import atom14_to_atom37
from openfold.np import residue_constants
import torch.nn.functional as F


def _dihedrals(X, eps=1e-7, return_angles=False):
    # First 3 coordinates are N, CA, C
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

    # Shifted slices of unit vectors
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]
    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, (1, 2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    phi, psi, omega = torch.unbind(D, -1)

    if return_angles:
        return phi, psi, omega

    # Lift angle representations to the circle
    D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    return D_features




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
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)

        phi, psi, omega = _dihedrals(final_atom_positions, eps=1e-7, return_angles=True)

        angles = torch.stack([phi.squeeze(), psi.squeeze(), omega.squeeze()], dim=1)
        final_atom_positions = final_atom_positions[:, :, self.ca_pos].squeeze(dim=0)


        # final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_distances = torch.norm(final_atom_positions - torch.roll(final_atom_positions, 1, 0), dim=-1)
        final_atom_distances[0] = 0

        # final_atom_angles =

        return torch.cat([final_atom_distances.unsqueeze(dim=-1), angles], dim=-1)


    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self