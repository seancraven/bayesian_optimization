"""
    Writing code for gp implemented in pytorch. 
"""
import torch
from torch import TensorType
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

    def forward(self, x: TensorType):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_guassian_process(
    g_p: GaussianProcessSurrogate,
    train_x: TensorType,
    train_y: TensorType,
    iter: int,
    verbose: bool = True,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = g_p(train_x, train_y, likelihood)

    model.train()

    opt = torch.optim.Adam(model.parameters())
    mll = gpytorch.ExactMarginalLogLikelihood(likelihood, model)



if __name__ == "__main__":
    train_x = torch.linspace(-5, 5, 10)
    train_y = torch.distributions.Normal(0, 5).sample(10)
    test_y = torch.linspace(-5, 5, 50)
