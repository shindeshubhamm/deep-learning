import torch
import torch.nn as nn


class Dropout(nn.Module):

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            x = x * mask / (1 - self.p)
        return x
    