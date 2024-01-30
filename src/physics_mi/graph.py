import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from .utils import unique


def add_plot_coords(gdf, layer_keys):
    """
    Adds plot coordinates to a graph dataframe (containing nodes) based on layer keys.

    Parameters:
    gdf (pd.DataFrame): The input graph dataframe.
    layer_keys (list): A list of layer keys.

    Returns:
    GeoDataFrame: The modified graph dataframe with plot coordinates added.
    """
    gdf["pos"] = None
    for lidx, lk in enumerate(layer_keys):
        layer_nodes = gdf.loc[gdf["layer"] == lk].index
        Y = np.linspace(-0.5, +0.5, len(layer_nodes) + 2)[1:-1][::-1]
        X = np.full(len(layer_nodes), float(lidx))
        coords = np.stack((X, Y)).T
        gdf.loc[layer_nodes, "pos"] = list(coords[:, None, :])
    return gdf


def nx_graph_from_pandas(gdf: pd.DataFrame, edf: pd.DataFrame):
    """
    Create a NetworkX graph from pandas DataFrames.

    Parameters:
    gdf (pandas.DataFrame): DataFrame containing node attributes.
    edf (pandas.DataFrame): DataFrame containing edge attributes.

    Returns:
    nx.DiGraph: NetworkX directed graph.
    """
    G = nx.from_pandas_edgelist(edf, "source", "target", edge_attr="weight", create_using=nx.DiGraph())

    # Add Node Attributes
    for _, row in gdf.iterrows():
        node = row.name
        attrs = row.to_dict()
        attrs["pos"] = attrs["pos"][0]
        nx.set_node_attributes(G, {node: attrs})

    return G


def plot_nx_graph(G):
    """
    Plot a networkx graph with weighted edges.

    Parameters:
    - G (networkx.Graph): The graph to be plotted.

    Returns:
    - None
    """

    max_width = 2.0

    pos = nx.get_node_attributes(G, "pos")
    edge_weights = nx.get_edge_attributes(G, "weight")
    scaled_weights = {
        (u, v): weight / max(edge_weights.values()) * max_width for (u, v), weight in edge_weights.items()
    }

    plt.figure(figsize=(15, 10))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="gray", node_size=10)

    # Draw edges with weights influencing the line thickness
    for (u, v), weight in scaled_weights.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight, edge_color="gray")


def create_output_layer_nodes(ios, output_lk):
    """
    Create nodes for the output layer.

    NOTE: we have to treat the output layer differently because it is in privileged basis

    Args:
        ios (dict): Dictionary containing input/output tensors for each layer.
        output_lk (str): Key for the output layer.

    Returns:
        list: List of dictionaries representing the nodes in the output layer.
    """
    output = ios[output_lk]
    num_classes = output.shape[1]
    indices = torch.arange(0, num_classes)
    one_hot = torch.eye(num_classes)[indices]
    nodes = [
        {
            "node": cidx,
            "layer": output_lk,
            "comp": one_hot[cidx],
            "norm": 1.0,
        }
        for cidx, _ in enumerate(output.transpose(1, 0))
    ]
    return nodes


def forward_pass(ios, lk_input, lidx, scaffold_model):
    """
    Perform a forward pass through the scaffold model.

    Args:
        ios (dict): Dictionary containing input tensors.
        lk_input (str): Key for the input tensor to be used.
        lidx (int): Input layer index.
        scaffold_model (nn.Module): The scaffold model.

    Returns:
        tuple: A tuple containing the input tensor and the output tensor.
    """
    linputs = ios[lk_input].clone().requires_grad_(True)
    out = scaffold_model(lidx, linputs)
    return linputs, out


def process_outputs(output_layer_acts, output_layer_nodes):
    """
    Process the output layer activations and nodes to compute the output component activations.

    Args:
        output_layer_acts (torch.Tensor): The activations of the output layer.
        output_layer_nodes (dict): The nodes of the output layer.

    Returns:
        torch.Tensor: The activations of the output components.

    """
    output_layer_comps = torch.stack(output_layer_nodes["comp"].tolist())
    output_layer_comps = output_layer_comps / torch.norm(
        output_layer_comps, dim=1, keepdim=True
    )  # normalise because we want to examine the upstream weights, not downstream
    output_layer_comp_acts = torch.einsum(
        "ij,kj->ik", output_layer_acts, output_layer_comps
    )  # mapping onto the output components
    return output_layer_comp_acts


def get_unique_gradients(component_acts, input_layer_acts):
    """
    Compute the unique gradients of the component activations with respect to the input layer activations.

    Args:
        component_acts (torch.Tensor): Tensor of component activations.
        input_layer_acts (torch.Tensor): Tensor of input layer activations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the unique input gradients and their corresponding norms.
    """
    loss = component_acts.mean()
    loss.backward(retain_graph=True)
    input_grads = input_layer_acts.grad.clone().detach()
    uq_input_grads, _, _, index = unique(input_grads, dim=0)
    uq_input_grads = uq_input_grads[
        torch.argsort(index)
    ]  # now in order of appearance so they can be matched with grads from other output nodes
    uq_input_grad_norms = torch.norm(uq_input_grads, dim=1)
    return uq_input_grads, uq_input_grad_norms


def generate_graph(ios, layer_keys, scaffold_model):
    """
    Generates a graph representation of the neural network by creating nodes and edges.

    Args:
        ios (dict): contains intermediate activations (as well as inputs and outputs)
        layer_keys (list): List of layer keys.
        scaffold_model: The scaffold model.

    Returns:
        node_df (DataFrame): DataFrame containing the nodes of the graph.
        edge_df (DataFrame): DataFrame containing the edges of the graph.
    """
    nodes = create_output_layer_nodes(ios, layer_keys[-1])
    edges = []

    node_df = pd.DataFrame(nodes)
    node_idx = node_df["node"].max() + 1

    reversed_layer_keys = layer_keys[::-1]
    reverse_indices = list(range(len(layer_keys) - 1))[::-1]

    # Iterate over pairs of layers (backwards)
    for lidx, lk_input, lk_output in zip(
        reverse_indices,
        reversed_layer_keys[1:],
        reversed_layer_keys[:-1],
    ):
        input_layer_acts, output_layer_acts = forward_pass(ios, lk_input, lidx, scaffold_model)
        output_layer_nodes = node_df.loc[node_df["layer"] == lk_output]
        output_layer_comp_acts = process_outputs(output_layer_acts, output_layer_nodes)

        # backprop from each node/component of the output layer
        for cidx, comp_acts in enumerate(output_layer_comp_acts.transpose(1, 0)):
            uq_input_grads, uq_input_grad_norms = get_unique_gradients(comp_acts, input_layer_acts)

            # store the gradients and their norms as nodes and edges
            for grad, norm in zip(uq_input_grads, uq_input_grad_norms):
                norm = float(norm)
                node = {"node": node_idx, "layer": lk_input, "comp": grad, "norm": norm}
                edge = {
                    "source": node_idx,
                    "target": output_layer_nodes["node"].iloc[cidx],
                    "weight": norm,
                }
                nodes.append(node)
                edges.append(edge)
                node_df = pd.DataFrame(nodes)
                node_idx += 1

            input_layer_acts.grad.zero_()

    edge_df = pd.DataFrame(edges)
    node_df.set_index("node", inplace=True)

    return node_df, edge_df
