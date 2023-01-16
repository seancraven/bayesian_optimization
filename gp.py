"""
    Writing code for gp implemented in pytorch. 
"""
import torch
from torch import Tensor
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

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_guassian_process(
    train_x: Tensor,
    train_y: Tensor,
    iterations: int,
    verbose: bool = True,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GaussianProcessSurrogate(train_x, train_y, likelihood)

    model.train()

    opt = torch.optim.Adam(model.parameters())
    mll = gpytorch.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(iterations):
        opt.zero_grad()
        out = model(train_x)
        loss = -mll(out, train_y)
        loss.backward()
        if verbose:
            print(f" Iteration: {i+1}, Loss: {loss.item()}")
        opt.step()

    return model, likelihood


def plot_gp(trained_model: models, likelihood, test_x: Tensor):
    trained_model.eval()
    likelihood.eval()
    with torch.no_grad():
        f_pred = trained_model(test_x)
        y_pred, f_mean = likelihood(f_pred), f_pred.mean
        upper, lower = y_pred.confidence_region()
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(test_x.numpy(), f_mean.numpy(), "b")
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    return f, ax


if __name__ == "__main__":
    train_x = torch.linspace(-5, 5, 10)
    train_y = torch.distributions.Normal(0, 5).sample((10,))
    test_x = torch.linspace(-5, 5, 50)
    model, likelihood = train_guassian_process(train_x[:, None], train_y, 20)
    fig, ax = plot_gp(model, likelihood, test_x)
    ax.plot(train_x, train_y, "k*")
    plt.show()
