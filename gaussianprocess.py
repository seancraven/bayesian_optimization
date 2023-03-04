import torch
import matplotlib.pyplot as plt
from typing import Tuple
import time

import visclassifier
from svm import dualSVM_adv
from data import train_x, train_y
from svm import computeK
import math
import imageio

torch.manual_seed(0)


class GuassianProcess:
    def __init__(
        self, kernel_type, kernel_param_1, noise=0.1, kernel_param_2: float = 1.0, log=False
    ):
        assert kernel_type in ["linear", "polynomial", "rbf", "poly"], "invalid kernel"
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param_1
        self.noise = noise
        self.kernel_param_2 = kernel_param_2
        self.log = log

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """Computes inverse kernel and stores training data."""
        if self.log:
            self.x = train_x.log10()
        else:
            self.x = train_x
        self.y = train_y
        self.kernel = self.kernel_param_2 * computeK(
            self.kernel_type, self.x, self.x, self.kernel_param
        )
        self.kernel_inv = (
            self.kernel + torch.eye(self.x.shape[0]) * self.noise**2
        ).inverse()

    def predict(
        self, test_x: torch.Tensor, offset: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and covariance tensors for the gp.

        Offset is the prior on the mean function. this is y=0.5 for error.
        """

        if self.log:
            test_x = test_x.log10()
        
        k_x_test = self.kernel_param_2 * computeK(
            self.kernel_type, test_x, self.x, self.kernel_param
        )
        k_test = self.kernel_param_2 * computeK(
            self.kernel_type, test_x, test_x, self.kernel_param
        )

        mean = offset + k_x_test @ (self.kernel_inv @ (self.y - offset))
        temp_mat = k_x_test @ self.kernel_inv @ k_x_test.T
        covariance = k_test - temp_mat
        
        return mean, covariance


def aquisition_functon(
    x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, iterations: int
):
    """uses lcb to find new point to evaluate at."""
    lcb = (mean - cov.diag().sqrt() * iterations).nan_to_num(0.5)
    
    plt.plot(lcb)
    plt.savefig(f"./img/lcb/lcb{iterations}.png")

    index = torch.argmin(lcb, dim=0)
    return x[index]


def plot_error_surrogate(
    mean,
    cov,
    hyperparam_data,
    hyperparam_error,
    search_grids: Tuple[torch.Tensor, torch.Tensor],
):
    c_grid, len_grid = search_grids
    points = c_grid.shape[0]

    std = cov.diag().sqrt()
    minus_one_sigma = mean - std
    

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(c_grid, len_grid, mean.reshape(-1, points), alpha=0.1)

    surf_sig = ax.plot_surface(
        c_grid, len_grid, minus_one_sigma.reshape(-1, points), alpha=0.4
    )
    scat = ax.scatter3D(
        hyperparam_data[:, 0],
        hyperparam_data[:, 1],
        hyperparam_error,
        "ko",
    )
    new_point = ax.scatter3D(
        hyperparam_data[-1, 0],
        hyperparam_data[-1, 1],
        hyperparam_error[-1],
        color="red",
        marker="o",
        s=50,
    )
    number_of_iterations = hyperparam_error.shape[0]
    # ax.view_init(30, (i*5)%360)
    ax.set_ylabel("Lengthscale Range")
    ax.set_xlabel("Slack Range")
    fig.savefig(f"./img/img/img_{number_of_iterations}.png", transparent=False)
    plt.close()

    frame = imageio.v2.imread(f"./img/img/img_{number_of_iterations}.png")

    return frame


def svm_bayes(
    surrogate: GuassianProcess,
    iterations: int,
    train_x: torch.Tensor,
    train_y,
    points=50,
    c_min=1e-4,
    c_max=1e2,
    len_min=1e-4,
    len_max=1e1,
    log=False,
):
    """
    Optimises svm using a guassian process with rbf kernel and lcb aquisition function.

    :params:
        surrogate: Instance of a guassian process.
        Iterations: Number of bayesian optimisatioins to run, starts with 16
        inital evaluations.
        points: The number of points in the search space
        c_min: Slack parameter min value.
        c_max: Slack parameter max value. 
        len_min: Inverse Lengthscale minimum value.
        len_max: Inverse Lengthscale maximum value.
        log: weather to search logspace in base 10
        
    """
    surrogate.log = log
    
    n = train_y.shape[0]
    index = torch.randperm(n)
    train_x = train_x[index]
    train_y = train_y[index]
    train_x, val_x = train_x[: -n // 2], train_x[-n // 2 :]
    train_y, val_y = train_y[: -n // 2], train_y[-n // 2 :]

    #
    base = 1
    if log:
        base = 10

    c_search = base * torch.linspace(c_min, c_max, points)
    inverse_lengthscale_search = base * torch.linspace(len_min, len_max, points)
    c_init = base * torch.linspace(c_min, c_max, 4)
    len_init = base * torch.linspace(len_min, len_max, 4)

    # c_search = torch.logspace(-3, 1, points)
    # inverse_lengthscale_search = torch.logspace(2, 4, points)

    len_grid = torch.stack([inverse_lengthscale_search] * points, dim=1).T
    c_grid = torch.stack([c_search] * points, dim=1)

    # c_grid changes every points values len_grid changes every value, all combinations of them are expressed.
    # print to see
    hyperparm_search = torch.stack(
        (c_grid.reshape(-1, 1), len_grid.reshape(-1, 1)), dim=1
    ).squeeze()

    hyperparam_error = []
    hyperparam_data = []
    
    
    # Gather intial points data
    for C in c_init:
        for len in len_init:
            svm = dualSVM_adv(
                train_x,
                train_y,
                "rbf",
                C=C,
                lmbda=len,
                mom=0.9,
                clip=True,
                num_epochs=50,
            )
            svm_val_error = (torch.sign(svm(val_x)) != val_y).float().mean().item()
            hyperparam_error.append(svm_val_error)
            hyperparam_data.append(torch.Tensor([C, len]))

    hyperparam_data = torch.stack(hyperparam_data, dim=0)
    hyperparam_error = torch.Tensor(hyperparam_error)

    assert hyperparam_data.shape[0] == hyperparam_error.shape[0]
    #    print("hyperparam error", hyperparam_error)
    frames = []
    surrogate.fit(hyperparam_data, hyperparam_error)
    for i in range(iterations):
        
        # if log the surrogate does all prediciction in logspace
        mean, cov = surrogate.predict(hyperparm_search)
        if i%10 == 0:
            print(mean.min().item()) 
        if log:
            frame = plot_error_surrogate(
                mean,
                cov,
                hyperparam_data.log10(),
                hyperparam_error,
                (c_grid.log10(), len_grid.log10()),
            )
        else:
            frame = plot_error_surrogate(
            mean, cov, hyperparam_data, hyperparam_error,(c_grid, len_grid)
            )
        
        frames.append(frame)

        new_hyperparam = aquisition_functon(hyperparm_search, mean, cov, i + 3)
        c_new = new_hyperparam[0]
        len_new = new_hyperparam[1]

        new_svm = dualSVM_adv(
            train_x,
            train_y,
            "rbf",
            C=c_new,
            lmbda=len_new,
            mom=0.6,
            clip=True,
            num_epochs=100,
        )
        new_error = (torch.sign(new_svm(val_x)) != val_y).float().mean().item()

        hyperparam_data = torch.cat([hyperparam_data, new_hyperparam[None, :]], dim=0)
        hyperparam_error = torch.cat(
            [hyperparam_error, torch.Tensor([new_error])], dim=0
        )

        surrogate.fit(hyperparam_data, hyperparam_error)
    #        print(f"Iteration: {i+3}")
    #        print(f"Validation Error: {new_error*100:.2f}%")

    
    
    best_settings_ind = torch.argmin(hyperparam_error)
    print("Best setting validation acc: ", hyperparam_error[best_settings_ind].item())
    best_settings = hyperparam_data[best_settings_ind]

    all_x = torch.cat((train_x, val_x), dim=0)
    all_y = torch.cat((train_y, val_y), dim=0)
    best_svm = dualSVM_adv(
        all_x,
        all_y,
        "rbf",
        C=best_settings[0],
        lmbda=best_settings[1],
        num_epochs=100,
        clip=True,
        mom=0.6,
    )
    return best_svm, frames


if __name__ == "__main__":
    gp_len = 0.005
    gp_mag = 0.05
    sur = GuassianProcess("rbf", gp_len, kernel_param_2=gp_mag)
    t0 = time.time()
    best_svm, frames = svm_bayes(sur, 60, train_x, train_y, points=80, log=False)
    t1 = time.time()
    print(f"Runtime : {t1-t0:.2f}")
    visclassifier.visclassifier(
        best_svm,
        train_x,
        train_y,
        save=f"./img/best_svm/gp_hyper_{gp_len}_{gp_mag}.png",
    )
    best_error = (torch.sign(best_svm(train_x)) != train_y).float().mean().item()
    print("best acc", best_error)
    imageio.mimsave(
        f"./img/gif/gif_len{gp_len}_mag{gp_mag}.gif",  # output gif
        frames,  # array of input frames
        fps=5,
    )
