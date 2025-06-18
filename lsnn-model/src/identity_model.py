import torch
import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        return x  # No processing — just passes the input directly