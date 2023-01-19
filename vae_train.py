import torch
import itertools
import numpy as np
from torch import permute
from torch import functional as f
from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torchvision import transforms
from torchvision import utils as vutils
from vae import *
from torch import autograd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Dont Edit
manual_seed = 1
torch.manual_seed(manual_seed)


dataroot = "../data/celeba/img_align_celeba/"

image_size = 64
image_channels = 3
latent_dim = 30
batch_size = 512

dataset = dset.ImageFolder(
    root=dataroot,
    transform=transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=10
)

if torch.cuda.is_available():
    device = "cuda"
else:
    raise Exception("No GPU")


def run_training(epochs: int):
    device = "cuda"
    enc = Encoder(image_size, image_channels, latent_dim)
    dec = Decoder(image_size, image_channels, latent_dim)
    enc.to(device), dec.to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()))
    for i in range(epochs):
        train_loss = 0
        for (X, _) in tqdm(dataloader):
            X = X.to(device)
            opt.zero_grad()
            loss = -elbo(enc, dec, X).mean()
            loss.backward()
            opt.step()
            train_loss += loss.item() * X.shape[0] * len(dataloader)
        tqdm.write(f"Epoch {i + 1}: Loss {train_loss:.3} \n")
    torch.save(enc.state_dict(), "./model/encoder/encoder.pt")
    torch.save(dec.state_dict(), "./model/decoder/decoder.pt")
    return enc, dec


if __name__ == "__main__":
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )
    # plt.show()

    enc, dec = run_training(30)
    enc = Encoder(image_size, image_channels, latent_dim)
    dec = Decoder(image_size, image_channels, latent_dim)
    enc.load_state_dict(torch.load("./model/encoder/encoder.pt"))
    dec.load_state_dict(torch.load("./model/decoder/decoder.pt"))
    enc.to("cpu")
    dec.to("cpu")
    plt.figure(figsize=(12, 4))
    plt.show()
    plt.figure(figsize=(12, 4))
    plt.axis("off")
    with torch.no_grad():
        dec.eval()
        enc.eval()
        mean, sigma = dec(enc(real_batch[0]).sample())
        out_dist = dist.Normal(mean, sigma)
        samples = out_dist.sample()
        print(samples.shape)
        plt.imshow(
            np.transpose(
                vutils.make_grid(samples[:64], padding=2, normalize=True),
                (1, 2, 0),
            )
        )
    plt.show()
