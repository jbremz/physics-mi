"""Putting the training into a script so I can automate a little more - see experiment 10"""

from pathlib import Path

import fire
import numpy as np
import torch
from physics_mi.net import SingleLayerNet
from physics_mi.utils import set_all_seeds
from torch import nn
from torch.optim import Adam
from tqdm.notebook import tqdm

MODELS = Path("models")


def main(seed, name):
    if seed is None:
        seed = np.random.randint(1, 2**32 - 1)
    set_all_seeds(seed)

    # ~~~~~~~~~~~~~~~~~~~~~~ DATA ~~~~~~~~~~~~~~~~~~~~~~

    eps = 0.5

    Y = torch.rand(10000)
    X = torch.empty(10000, 2)
    X[:, 0] = Y / (torch.rand(10000) * (1 - eps) + eps)
    X[:, 1] = Y / X[:, 0]

    # need to randomly swap x1 and x2 so that they're identically distributed - can do this because their product is commutative
    mask = torch.rand(10000) < 0.5
    swap_vals = X[:, 0][mask]
    X[:, 0][mask] = X[:, 1][mask]
    X[:, 1][mask] = swap_vals

    assert torch.allclose(X[:, 0] * X[:, 1], Y)

    Y = Y[:, None]

    s_inds = np.random.permutation(range(X.shape[0]))  # shuffled indices

    X_train = X[s_inds[:8000]]
    Y_train = Y[s_inds[:8000]]
    X_valid = X[s_inds[8000:]]
    Y_valid = Y[s_inds[8000:]]

    X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape

    # ~~~~~~~~~~~~~~~~~~~~~~ TRAIN ~~~~~~~~~~~~~~~~~~~~~~

    N = 4000  # number of epochs

    model = SingleLayerNet(use_act=True, dim=2)
    loss_func = nn.MSELoss()
    optimiser = Adam(model.parameters(), lr=1e-3)
    log = []

    for i in tqdm(range(N)):
        log_sample = {}

        # Training update
        model.train()
        model.zero_grad()
        Y_hat = model(X_train)
        loss = loss_func(Y_hat, Y_train)
        log_sample["train_loss"] = float(loss.detach())
        loss.backward()
        optimiser.step()

        # Validation set
        model.eval()
        Y_hat = model(X_valid)
        loss = loss_func(Y_hat, Y_valid)
        log_sample["valid_loss"] = float(loss.detach())

        log.append(log_sample)  # leaving the logging in for potential future use

    torch.save(model.state_dict(), MODELS / f"{name}-{seed}.pth")


if __name__ == "__main__":
    fire.Fire(main)
