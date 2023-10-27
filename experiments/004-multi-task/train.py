import logging
import pickle
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib import pyplot as plt
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
            LinearLayer(hidden_dim, output_dim, use_act=False),
        )

    def forward(self, x):
        return self.layers(x)


def get_valid_loss(model, loss_func):
    model.eval()

    with torch.inference_mode():
        out = model(X_valid)

    return loss_func(out, Y_valid)


def capture_intermediate_outputs(model, input_tensor):
    """
    Captures the intermediate outputs of a PyTorch model.

    Args:
        model: A PyTorch model.
        input_tensor: A PyTorch tensor of shape (batch_size, *).
    Returns:
        A dictionary mapping layer names to intermediate outputs.
    """
    intermediate_values = OrderedDict()

    def hook_fn(module, input, output, name):
        if isinstance(module, nn.Linear):
            weight = module.weight
            input_value = input[0]
            intermediate_output = input_value.matmul(weight.t())
            intermediate_values[f"{name}.weight"] = intermediate_output
            intermediate_values[f"{name}.bias"] = output
        else:
            intermediate_values[name] = output

    hooks = []
    for name, layer in model.named_modules():
        hook = layer.register_forward_hook(
            lambda module, input, output, name=name: hook_fn(module, input, output, name)
        )
        hooks.append(hook)

    with torch.inference_mode():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    filtered_values = {k: t for k, t in intermediate_values.items() if len(k.split(".")) > 2}
    return filtered_values


# a function that uses capture_intermediate_outputs to give a nice printed summary of the model outputs and parameters
def print_model_summary(model, input_tensor):
    intermediate_values = capture_intermediate_outputs(model, input_tensor)

    for k, v in intermediate_values.items():
        print(f"{k}: {v}")


repeats = 100

for repeat in tqdm(range(repeats)):
    seed = np.random.randint(1, 2**32 - 1)
    set_all_seeds(seed)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Number of samples
    n_samples = 10000

    # Epsilon value
    eps = 0.5

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

    s_inds = np.random.permutation(range(X.shape[0]))  # shuffled indices

    X_train = X[s_inds[:8000]]
    Y_train = Y[s_inds[:8000]]
    X_valid = X[s_inds[8000:]]
    Y_valid = Y[s_inds[8000:]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~

    n_epochs = 4000  # number of epochs

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

    N = 5
    pairs = np.concatenate(np.stack(np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))).T)
    pairs = pairs.repeat(2, axis=0).reshape(-1, 4)
    outputs = capture_intermediate_outputs(model, torch.as_tensor(pairs).float())

    res["valid_loss"] = get_valid_loss(model, loss_func)
    res["outputs"] = outputs
    res["model"] = model.state_dict()
    res["seed"] = seed

    fp = f"experiments/004-multi-task/results/{seed}.pkl"

    with open(fp, "wb") as f:
        pickle.dump(res, f)


logger.info("Completed all training runs")
