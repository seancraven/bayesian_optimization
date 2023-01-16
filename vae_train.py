import torch
from torch import nn
from torch import functional as f
from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torchvision import transforms
from torchvision import utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Dont Edit
manual_seed = 1
random.seed(manual_seed)
torch.manual_seed(manual_seed)


dataroot = "data/celeba"

image_size = 64
image_channels = 3
latent_dim = 10
batch_size = 256

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

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
if torch.cuda.is_available():
    device = "cuda"
else:
    "Fuckup"

if __name__ == "__main__":
