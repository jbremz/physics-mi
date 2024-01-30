# Computational graph

I'm now in the (exciting) position to be able to build a computational graph give one of these multi-task MLPs that I've trained (I think). I'm fairly confident at this point that there is orthogonal (independent) task processing happening internally in the network. I hoping that this next piece of work will convert this isolated understanding into a more understandable and useful representation, a graph.

I'm also quite excited to show some pretty pictures.

Dream goal eventually here is to then apply this technique to a very basic network trained on e.g. MNIST to hopefully reveal some structure. But that's getting ahead of myself.

## Vague thoughts

- I wonder if this would help in detecting/understanding adversarial examples better? i.e. we could look at the typical subgraph that's activated by examples in the dataset and unusually activated subgraphs might be finding adversarial routes through the network? 🤔
- I have a feeling that non-zero'd activation functions (like GELU) would break this unique gradients thing I'm doing? But still I suppose in those situations you might have to do clustering as opposed to finding unique components.
- Layerwise Relevance Propagation seems to be a similar method to this but I suppose is more interested in the input space than what's going on inside the network
- Just found this paper: [_Interpreting Neural Networks through the Polytype Lens_](https://www.lesswrong.com/posts/eDicGjD9yte6FLSie/interpreting-neural-networks-through-the-polytope-lens) which is very relevant and provides a nice description of a lot of the things I've been grappling with :) I have a strong feeling that my gradient splintering is closely related to if not a direct example of polytopes. They talk about the difficulties in interpretability this produces. I'm still finding that I'm able to extract useful structure from the gradients in terms of task separation, so let's see if I can continue with that without worrying too much about the _complete_ picture yet 😅

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

One thing I'm worrying about is whether we need to calculate the cross output component dot product on an example by example basis or whether it suffices to use an all vs. all on the unique gradients? It might be that for a particular example some gradients just aren't relevant/accurately applied. **Update:** I did a test and found that the gradients backpropped from different outputs always appear together, that's to say there are the same number of unique gradients for each separate output component as there are unique _combinations_ of gradients for all the output components combined (sorry there's definitely a much cleaner way of putting that). Probably worth checking and understanding, but this is great from a computational point of view because it means that we:

- don't need to do some fiddly dot products across every single example in the dataset
- even better, only need to do $n-1$ dot products per input node for $n$ output nodes (as each unique gradient for each output node only needs to be dotted with its equivalent for the other output nodes as it never appears in any other context)

I'm getting a better intuition now for the fact that this computation graph will only represent a combination of all the possible paths for a data example. I'm keen to examine how much of the computational graph is activated for each data example.

The gradient splintering happened even though I was back-propagating layer by layer and perhaps this makes sense. I can see this being an issue when thinking about scaling this method to deeper networks. Maybe there's some clustering that would be required. This feels like it potentially has links with superposition and scaling sparse autoencoders.

What I'm struggling with right now is this idea of gradient uniqueness. It becoming clear that whilst the dendrogram-like graph structure explodes exponentially as we increase depth, the actual possible paths as a subset of the total paths dislpayed are many fewer. In fact, for each input layer, there are only as many paths as there are unique gradients _for an individual node_ in the output layer. For example, if we back propagate from layer1 to layer0 and there are 10 nodes in layer1, with 23 unique gradients per output node in layer0, that _doesn't_ actually leave us with $10 \times 23 = 230$ entry points in layer0 because in reality, only one of the 23 paths is being used at any time. That's to say, the required gradients to produce change in the downstream nodes always appear together. I need to find some way of representing this in my graph structure.

### Results

I produced more clean results on task separation here which is cool. The thing I really struggled with was understanding how I should approach inter-layer connectivity. This was straightforward when I could use back propagation from individual downstream nodes, but intra-task and inter-node similarity in the gradients was proving to be a headache for fitting this network idea my idea of a computational graph. I ended up thinking that perhaps a computational graph wasn't such a good fit for neural networks, it wouldn't express the full picture.

Instead, where I've had most success is in clustering independent task features on a layer by layer basis, pulling out that kind of structure. My idea now is to continue on this line of enquiry and apply these same techniques to inputs that are partially independent i.e. there is logic for their combination.

First I'll tidy up the back-propagation code 🙃 and then I'll move on to a simple toy experiment to start task mixing.

## `002-tidy-backprop-code`

That was painful but I managed it. The code is still not _really_ clean but I'm finding it hard to make it that way with this graph stuff. Often one needs to keep track of objects between operations making modularising into functions tricky. Equally, the OOP approach doesn't feel so nice here when we want to do big parallel operations across nodes (really we want to be doing that with pytorch tensors as opposed to for loops). Anyway, there's probably some way of improving this but this is good enough for now.

## `003-simple-task-mixing`

I'm hoping now with my cleaner code, I can now more efficiently examine other tasks with this backprop method.

This first task I _think_ will involve a very trivial task mixing. There will be three inputs:
- $x_1$ - one number to multiply with...
- $x_2$ - another number
- $g$ - a gate

The logic is follows (in pseudo code):
```python
if g == 1:
    return x1 * x2
if g == 0:
    return 0
```
This way, we'll hopefully see that `g` _has_ to mix into the output at some point.

### Results

...are unclear. I've realised my tasks are commutative so there could be all sorts of mixing going on at any stage. Time to try something less commutative:

## `004-less-simple-task-mixing`

I coded up this:
$$
\text{Step 1:} \quad f(x, y) = x - y \\
\text{Step 2:} \quad g(f, z) = \frac{f}{z} = \frac{x - y}{z}
$$

There's definitely redundancy in the gradients that I can see in all the repeating patterns. It's where clustering would probably be a good idea...

Still, I have another idea of what I could try.

## `005-soft-mixing`

I might return to our parallel multiplication problem and see what happens if I introduce a bit of "soft" mixing. That is, some form of mixing that I can tone down smoothly to the independent task again. I suggest mixing each outputs, so weighting each task output with a small amount of the other task output.

Hopefully, this would allow us to studying the limiting cases more easily.