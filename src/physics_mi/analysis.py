import warnings
from collections import OrderedDict
from math import ceil

import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from torch import nn

warnings.filterwarnings(
    "ignore",
    "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
)


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


def plot_neuron_contributions(model, N=5):
    """Plots the contribution of each neuron to the final output.

    NOTE: this ignores the effect of the bias
    """

    eps = 0.05
    pairs = np.concatenate(np.stack(np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))).T)

    outputs = capture_intermediate_outputs(model, torch.as_tensor(pairs).float())
    lws = model.state_dict()["layers.1.linear.weight"]
    acts = outputs["layers.0.act"]
    acts = acts * lws[0, :]  # linear weighting now (I know they're not technically activations anymore)

    fig = plt.figure(figsize=(12, 10))  # Increase the width to make space for the colorbar

    for i in range(acts.shape[1]):
        ax = fig.add_subplot(4, 4, i + 1)
        col = acts[:, i]
        activated_mask = acts[:, i].abs() > eps  # really trying to focus on the activated areas
        activated_mask = torch.ones(col.shape, dtype=bool)
        sc = ax.scatter(
            pairs[activated_mask, 0],
            pairs[activated_mask, 1],
            c=col[activated_mask],
            cmap="bwr",
            s=50,
            norm=plt.Normalize(vmin=-1, vmax=1),
        )

        ax.set_title(f"Neuron {i+1}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Position the colorbar on the right of the last subplot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax)

    bias = float(model.state_dict()["layers.1.linear.bias"][0])

    fig.suptitle(f"Output value at layers.1.linear.weight\n(pre addition of {bias:.2f} bias)")
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rectangle in which to fit the subplots


def plot_model_breakdown(
    x1,
    x2,
    model,
    point_selector,
    ax=None,
    legend=True,
    color=None,
    product_color=False,
    lines=True,
):
    """
    Plots the breakdown of a model's output into its intermediate components.

    Args:
        x1: The first input value.
        x2: The second input value.
        point_selector: A dictionary mapping layer names to booleans. If a layer name is in the dictionary, then the corresponding point will be plotted.
        ax: A matplotlib axes object.
        legend: Whether to show the legend.
        color: The color of the points.
        product_color: Whether to color the points by the product of x1 and x2.
        lines: Whether to draw lines between the points.
    Returns:
        The matplotlib axes object.
    """
    if not ax:
        fig = plt.figure()
        ax = plt.axes()

    input = (x1, x2)

    if not color and product_color:
        val = x1 * x2
        cmap = cm.plasma
        val = val**0.5  # looks better on the unit square with some scaling
        color = cmap(val)

    x = capture_intermediate_outputs(model, torch.tensor([x1, x2]).float())
    dim = list(x.values())[1].shape[0]
    x = {k: t.tolist() for k, t in x.items()}
    x = {**{"input": list(input)}, **x}
    x = {
        k: (t + (dim - len(t)) * [0.0] if len(t) < dim else t) for k, t in x.items()
    }  # pad with 0s for input and near-output values

    markers = {k: "o" for k, _ in x.items()}
    markers["input"] = "v"
    markers[list(x.keys())[-1]] = "x"

    for point_name, point in x.items():
        if point_selector[point_name]:
            ax.scatter(
                *point,
                alpha=1.0,
                label=point_name,
                color=color,
                marker=markers[point_name],
            )

    x_arr = np.array(list(x.values()))

    if lines:
        ax.plot(*x_arr.T, lw=0.5, color=color)
    ax.set(xlabel="component 1", ylabel="component 2")
    if legend:
        ax.legend()


def get_colors(N):
    cmap = plt.get_cmap("turbo", N)
    colors = []
    for i in range(cmap.N):
        colors.append(cmap(i))
    return colors


def get_default_point_selector_sets():
    subsets = []
    for i in range(6):
        subset = {
            "input": False,
            "layers.0.linear.weight": False,
            "layers.0.linear.bias": False,
            "layers.0.act": False,
            "layers.1.linear.weight": False,
            "layers.1.linear.bias": True,
        }
        subset[list(subset.keys())[i]] = True
        subsets.append(subset)
    subsets[0]["input"] = True
    return subsets


def plot_subsets(model, axes=None, subsets=None):
    """Plots the breakdown of a model's output into its intermediate outputs."""
    if subsets is None:
        subsets = get_default_point_selector_sets()

    ncols = 2
    nrows = ceil(len(subsets[0]) / ncols)

    if axes is None:
        fig = plt.figure(figsize=(10, nrows * 5.5))

    N = 5
    pairs = np.concatenate(np.stack(np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))).T)

    titles = list(subsets[0].keys())[:-1]

    for i, (subset, title) in enumerate(zip(subsets, titles)):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_model_breakdown(
            *pairs[0],
            model=model,
            ax=ax,
            product_color=True,
            point_selector=subset,
            lines=False,
        )
        for pair in pairs[1:]:
            plot_model_breakdown(
                *pair,
                model=model,
                ax=ax,
                legend=False,
                product_color=True,
                point_selector=subset,
                lines=False,
            )
        ax.vlines(0, -2, 2, color="gray", ls="--", lw=0.5)
        ax.hlines(0, -2, 2, color="gray", ls="--", lw=0.5)
        ax.set_title(title)
        ax.set_aspect("equal")


def map_acts(comps, acts):
    """
    Map activations to components to determine the level of activation along each component using PyTorch tensors.

    This function performs a transformation that measures how much each activation vector in 'acts'
    aligns with a set of given components in 'comps'. It multiplies each component with the
    activations and sums the results, effectively projecting the activations onto the space defined
    by the components.

    Parameters:
    comps (torch.Tensor): A 2D tensor where each row represents a component. These components
                          are akin to basis vectors in a certain space.
    acts (torch.Tensor): A 2D tensor of activations. Each row corresponds to an activation vector
                         that will be mapped onto the components.

    Returns:
    torch.Tensor: A 2D tensor where each row represents an activation vector from 'acts'
                  transformed into the component space defined by 'comps'. The output dimensions
                  are determined by the number of components and the number of activation vectors.
    """
    return torch.einsum("ij,kj->ki", comps, acts)


def get_sims(node_df, layer_key):
    comps = torch.stack(node_df.loc[node_df["layer"] == layer_key, "comp"].tolist())
    comps = comps / torch.norm(comps, dim=1, keepdim=True)
    sims = torch.einsum("ij,kj->ik", comps, comps)
    return sims


def plot_similarity_matrix(sims):
    """
    Plots a similarity matrix.

    Parameters:
    sims (numpy.ndarray): The similarity matrix to be plotted.

    Returns:
    None
    """
    plt.imshow(sims)
    # Set the colormap to something diverging
    cmap = plt.get_cmap("coolwarm")

    # Create a centered normalization
    norm = mcolors.CenteredNorm(vcenter=0, halfrange=abs(sims).max())

    # Plot the data with the divergent colormap and centered normalization
    plt.imshow(sims, cmap=cmap, norm=norm)
    plt.colorbar()  # Add color bar legend
