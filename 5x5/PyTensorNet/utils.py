import numpy as np
import torch
#2
def init_tensor(shape, bond_str, init_std):

    #std = 1e-9
    tensor = init_std * torch.randn(shape)

    return tensor


