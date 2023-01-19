import torch
from torch.nn import functional as f
from torch import nn
from torch import distributions as dist


class Encoder(nn.Module):
    def __init__(self, image_size, image_chan, latent_dim):
        super().__init__()
        self.image_size = image_size

        self.conv_1 = nn.Conv2d(image_chan, 64, 3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(128)

        self.conv_3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(256)

        self.lin_1 = nn.Linear(256 * (image_size // 2**3) ** 2, 500)

        self.lin_2_mean = nn.Linear(500, latent_dim)
        self.lin_2_cov = nn.Linear(500, latent_dim)

    def forward(self, x):
        x = f.relu(self.bn_1(self.conv_1(x)))
        x = f.max_pool2d(x, kernel_size=2)
        x = f.relu(self.bn_2(self.conv_2(x)))
        x = f.max_pool2d(x, kernel_size=2)
        x = f.relu(self.bn_3(self.conv_3(x)))
        x = f.max_pool2d(x, kernel_size=2)
        x = x.reshape((x.shape[0], -1))
        assert x.shape[1] == 256 * (self.image_size // 8) ** 2, f"{x.shape}"
        x = f.relu(self.lin_1(x))
        mean = self.lin_2_mean(x)
        cov = torch.clamp_min_(self.lin_2_cov(x), 0.01)
        return dist.Normal(mean, cov)


class Decoder(nn.Module):
    def __init__(self, image_size, image_chan, latent_dim):
        super().__init__()
        self.image_size = image_size
        self.lin_1 = nn.Linear(latent_dim, (image_size // 2**4) ** 2)

        self.convt_1 = nn.ConvTranspose2d(
            1, 128, 3, stride=2, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(128)

        self.convt_2 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=0, bias=False
        )
        self.bn_2 = nn.BatchNorm2d(64)
        self.convt_3 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=0, bias=False
        )
        self.bn_3 = nn.BatchNorm2d(32)

        self.convt_4_mean = nn.ConvTranspose2d(
            32, 3, 3, stride=2, output_padding=1, bias=False
        )
        self.convt_4_var = nn.ConvTranspose2d(
            32, 3, 3, stride=2, output_padding=1, bias=False
        )

    def forward(self, z):
        z = f.relu(self.lin_1(z))
        z = z.reshape(
            (z.shape[0], 1, self.image_size // 16, self.image_size // 16)
        )
        z = f.relu(self.bn_1(self.convt_1(z)))
        z = f.relu(self.bn_2(self.convt_2(z)))
        z = f.relu(self.bn_3(self.convt_3(z)))
        mean = self.convt_4_mean(z).exp()
        sigma = self.convt_4_var(z).exp()
        mean = mean.reshape(mean.shape[0], 3, self.image_size, self.image_size)
        sigma = sigma.reshape(
            mean.shape[0], 3, self.image_size, self.image_size
        )

        return mean, sigma


def elbo(enc, dec, x):
    """
    This returns the mean of the batch wise elbo.
    """
    p_z = dist.Normal(0, 1)
    q_zx = enc(x)
    z = q_zx.rsample()
    x_hat, sigma_hat = dec(z)
    # print(sigma_hat)
    p_z = p_z.log_prob(z).sum(dim=-1)
    q_z = q_zx.log_prob(z).sum(dim=-1)
    p_x = torch.clamp(
        (-((x - x_hat) ** 2) / (2 * sigma_hat**2)), min=-10e3, max=10e3
    ).sum(dim=(1, 2, 3))
    return p_x + p_z - q_z
