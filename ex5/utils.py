import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BatchNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.theta_mu = torch.zeros(num_channels)
        self.theta_sigma = torch.ones(num_channels)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            if self.running_mean is None:
                self.running_mean = mean
                self.running_var = var
                var = x.var(dim=0)
            else:
                self.running_mean = 0.9 * self.running_mean + 0.1 * mean
                self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            if self.running_mean is None:
                raise TypeError()
            else:
                mean = self.running_mean
                var = self.running_var

        x = (x - mean) / (var + self.eps).sqrt() * self.theta_sigma + self.theta_mu

        return x
