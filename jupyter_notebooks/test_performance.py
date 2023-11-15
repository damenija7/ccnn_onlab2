import os
import pickle

import pydssp
import torch


for pdb in ['../utils/2zta.pdb']:
    # read pdb file
    coord = torch.tensor(pydssp.read_pdbtext(open(pdb, 'r').read()))
    # main calcuration
    res = pydssp.assign(torch.tensor(pydssp.read_pdbtext(open(pdb, 'r').read())))
    # write file or STDOUT