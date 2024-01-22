# Computational graph

I'm now in the (exciting) position to be able to build a computational graph give one of these multi-task MLPs that I've trained (I think). I'm fairly confident at this point that there is orthogonal (independent) task processing happening internally in the network. I hoping that this next piece of work will convert this isolated understanding into a more understandable and useful representation, a graph.

I'm also quite excited to show some pretty pictures.

Dream goal eventually here is to then apply this technique to a very basic network trained on e.g. MNIST to hopefully reveal some structure. But that's getting ahead of myself.

## Vague thoughts

- I wonder if this would help in detecting/understanding adversarial examples better? i.e. we could look at the typical subgraph that's activated by examples in the dataset and unusually activated subgraphs might be finding adversarial routes through the network? 🤔
- I have a feeling that non-zero'd activation functions (like GELU) would break this unique gradients thing I'm doing? But still I suppose in those situations you might have to do clustering as opposed to finding unique components.

## `001-inter-layer-backprop-graph`

One thing I learnt from `006-layer-correlation/007-expand-backprop-analysis` was that by the time you had back propagated from the outputs to the input, there was some kind of gradient splintering, often leaving me with 100+ unique gradients for each task output. This makes sense on an intuitive level in that the network is essentially removing information and compressing the inputs into a computed form(?)

However, in this case, I would like to return to back propagating one layer at a time because:

- this splintering could get quite extreme for more realistic (deeper) networks whereas I _think_ there should be limits to how much splintering happens between two layers
- there could also be various pieces of sub computation that the network is doing that _don't_ make their way to the output (maybe they get masked by a ReLU). It might be interesting to study these, and that would require adjacent layer back propagation.
- adjacent layer backprop seems to be a more agnostic way to approach this (in that we're maybe making fewer assumptions?)

Here I'm going to

- take my multitask model from before
- extract adjacent layer gradients using the mapping method with the network scaffold I suppose with _normalised_ gradients from the following layer (so as to keep the edge-weights layer-wise, not output-wise - the idea is to create some kind of flow)
- extract the intralayer gradients by dotting between gradients for each of the following layer's components. This is a bit difficult to explain but hopefully in code it'll make more sense.
- create the edge weights accordingly

Let's see how I go.
