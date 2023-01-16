"""
    Writing code for gp implemented in pytorch. 
"""
import torch
import gpytorch
from gpytorch import kernels, models
import matplotlib.pyplot as plt


class GaussianProcessSurrogate(models.ExactGP):
    """
    Guassian process proxy for black box optimisation
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            kernels.MaternKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
