import torch
from torch.nn import functional as f
from torch import nn


class Encoder(nn.Module):
    def __init__(self, image_size, image_chan, latent_dim):
        super().__init__()
        self.image_size
        self.conv_1 = nn.Conv2d(image_chan, 64, 3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn_2 = nn.BatchNorm2d(128)
        self.lin_1 = nn.Linear(128 * image_size // 2**2, 100)
        self.lin_2_mean = nn.Linear(100, latent_dim)
        self.lin_2_cov = nn.Linear(100, latent_dim)

    def forward(self, x):
        x = f.relu(self.bn_1(self.conv_1(x)))
        x = f.max_pool2d(x)
        x = f.relu(self.bn_2(self.conv_2(x)))
        x = f.max_pool2d(x)
        assert x.shape[0] == 128 * self.image_size // 2**2
        x = f.relu(self.lin_1(x))
        mean = self.lin_2_mean(x)
        cov = self.lin_2_cov(x).exp()
        return mean, cov


class Decoder(nn.Module):
    def __init__(self, image_size, image_chan, latent_dim):
        super().__init__()
