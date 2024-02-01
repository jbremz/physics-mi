# modified from `006-layer-correlation/train.py` to:
#   - add tasking mixing

import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from physics_mi.utils import set_all_seeds

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Generate Y values
def generate_Y(n_samples):
    return torch.rand(n_samples)


# Generate X values based on Y
def generate_X(Y, eps):
    X = torch.empty(len(Y), 2)
    X[:, 0] = Y / (torch.rand(len(Y)) * (1 - eps) + eps)
    X[:, 1] = Y / X[:, 0]

    # Randomly swap x1 and x2
    mask = torch.rand(len(Y)) < 0.5
    swap_vals = X[:, 0][mask]
    X[:, 0][mask] = X[:, 1][mask]
    X[:, 1][mask] = swap_vals

    return X


def generate_data(n_samples=10000, eps=0.5, λ=0.0):
    # Initial generation
    Y1 = generate_Y(n_samples)
    X1 = generate_X(Y1, eps)

    # Ensure they are statistically independent by generating new Y and X values
    Y2 = generate_Y(n_samples)
    X2 = generate_X(Y2, eps)

    # Stack X1 and X2 to get the desired shape
    X = torch.cat((X1, X2), dim=1)

    # Stack Y1 and Y2 for the desired shape
    Y = torch.stack((Y1, Y2), dim=1)

    # Validate the relationship
    assert torch.allclose(X[:, 0] * X[:, 1], Y[:, 0])
    assert torch.allclose(X[:, 2] * X[:, 3], Y[:, 1])

    # Mix the tasks
    nY0 = (1 - λ) * Y[:, 0] + λ * Y[:, 1]
    nY1 = λ * Y[:, 0] + (1 - λ) * Y[:, 1]
    Y = torch.cat((nY0.unsqueeze(1), nY1.unsqueeze(1)), dim=1)

    s_inds = np.random.permutation(range(X.shape[0]))  # shuffled indices

    X_train = X[s_inds[:8000]]
    Y_train = Y[s_inds[:8000]]
    X_valid = X[s_inds[8000:]]
    Y_valid = Y[s_inds[8000:]]

    return X_train, Y_train, X_valid, Y_valid


class LinearLayer(nn.Module):
    def __init__(self, in_feats, out_feats, use_act=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(in_feats, out_feats)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act

    def forward(self, x):
        x = self.linear(x)
        if self.use_act:
            x = self.act(x)
        return x


class Net(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            LinearLayer(input_dim, hidden_dim, use_act=True),
            LinearLayer(hidden_dim, hidden_dim, use_act=True),
            LinearLayer(hidden_dim, output_dim, use_act=False),
        )

    def forward(self, x):
        return self.layers(x)


def get_valid_loss(model, loss_func):
    model.eval()

    with torch.inference_mode():
        out = model(X_valid)

    return loss_func(out, Y_valid)


if __name__ == "__main__":
    repeats = 5
    λs = np.linspace(0, 0.5, 11)
    rows = []
    for λ in λs:
        logger.info(f"λ = {λ}")
        for repeat in tqdm(range(repeats)):
            seed = np.random.randint(1, 2**32 - 1)
            set_all_seeds(seed)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~ DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

            X_train, Y_train, X_valid, Y_valid = generate_data(10000, eps=0.5, λ=λ)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~

            n_epochs = 1000  # number of epochs

            model = Net(input_dim=4, hidden_dim=16, output_dim=2)
            loss_func = nn.MSELoss()
            optimiser = Adam(model.parameters(), lr=1e-2)
            log = []

            for i in range(n_epochs):
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

                log.append(log_sample)

            df = pd.DataFrame(log)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~ EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~~~

            res = {}

            res["valid_loss"] = float(get_valid_loss(model, loss_func))
            res["model"] = model.state_dict()
            res["seed"] = seed
            res["λ"] = λ

            rows.append(res)

    logger.info("Completed all training runs")

    rdf = pd.DataFrame(rows)
    rdf.to_pickle("results/results.pkl")
