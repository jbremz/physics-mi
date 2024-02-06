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

### Results

Turned out pretty nice. I made a nice grid plot with a number of trained networks with different values of $\lambda$ (the task mixing parameter I made). I created a task orthogonality score for the first layer and plotted it against $\lambda$ and you can see a nice decrease in task orthogonality as $\lambda$ increases, as expected.

I suppose you could use this to generate an orthogonality score between any nodes in the same layer by partitioning the components of the previous layer correctly (according to the gradient dendrogram structure).

### What now?

What I've been able to do here is to pull out components that the network is using internally for different tasks. There remain some questions:
1. if we make the task more difficult (i.e. not a single mathemetical operation), can we start to discern subtasks, or is this method not expressive enough for that?
1. if we _can_ discern independent subtasks in the network, how can we then cluster them / put them in context?
1. if we can do the above, can we move towards our unsupervised embedder of network components?
1. there's the lingering question of how superposition might play a part in all of this... 🙈

Ideas on what I'd like to do next:
1. With 1. (above) in mind, I'd like to try to increase the task difficulty. Some ideas:
    1. MNIST - hey I know that's a huge jump and _could_ also have all sorts of superposition trouble _but_ I think there might be benefit in at least trying. It might raise some issues that happen with real-life datasets and larger models (I imagine I'll have to make the model at least a little larger).
    1. some nasty long mathematical formula idk 🤷
1. I've been wanting to understand all these splintered gradients a little better and I've realised I can do a nice plot to explore this. It would involve looking at the input space for a single task in the multitask setting (ignoring the inputs for the other task given that we've proven there is negligible interference) and plotting the vector field of the gradients across a unit square input (with fun little arrows). I think this would be straightforward to do and might bring up some patterns that were otherwise not so obvious...

## `006-vector-field`

I'm going with the latter option above because it should be pretty quick and is hopefully illuminating.

### Plan

1. Take trained model
1. Calculate gradients for each grid point of a generated unit square
1. Plot this field with little arrows 💘

### Results

- pretty good!
- this is one of the few cases where I realise I'm able to get a theoretical prediction of what the gradients should look like and how the network is optimised to mimic them 😊
- seems to be some fairly clear evidence in support of the polytope view of neural networks (maybe kind of seems obvious in retrospect...)

### Where does this leave us?

- well it seems to leave us right back in piecewise-linear land with probably the clearest idea of it yet (hehe it took me this long)
- suggests that the network is indeed simply partitioning up the input space into many little regions that each act linearly on the output. I'm guessing that deeper/wider networks will just be able to partition the input space into smaller linear regions and therefore model non-linear functions (e.g. like this one) with higher fidelity?
- this is still the independent and single task view though and I'm interested to see how one could:
    - use this perspective to classify/embed parts of the network that lie between two points in a network (i.e. pairings of activation inputs and outputs with varying amounts of separation). Just in the same way that I was able to see that the gradients of my network mimicked the theoretically optimal gradients of $y=x_1x_2$, perhaps we could train a system to do the same thing? It would look at some input data gradients and make out the piecewise linear regions and hopefully what they are modelling. Depending on where we put our "electrodes" we could measure the effect of the layers with increasing/decreasing granularity e.g. just look at the part of the network that deals with the region of high $x_1$ and low $x_2$.
    - study the task mixing aspect... 

### Thinking more about classification

In the back of my head I've still been wondering how I might train an unsupervised system to embed different parts of the network. Here's my latest idea (based on the thoughts above):
- provide:
    - intermediate activations (a sampling of the space)
    - the gradients produced between these intermediate activations and some component later on in the layer (how far later depends on the granularity of your study)
- train it with a cloze task. Given _all_ of the inputs and only some of the gradients the model needs to fill in the missing gradients, and in doing so hopefully understand the underlying function it represents. This amounts to predicting piecewise-linear sections of a larger function.
- can extract an embedding which would hopefully represent some information about the underlying function being modelled by that part of the network (on that data)

How I might test the usefulness of this method:
1. Train a variety of different (single) tasks - probably different mathematical formulae for now. Multiple trains for each task so I build up my "dataset" of different models e.g. 10 tasks x 50 models/task = 500 models. 
1. Partition my dataset traditionally so that each task appears in both the training and validation sets but _importantly_ with different seeded trained models e.g. 10 models per task are in the validation set
1. Generate new input data each batch and run it through the models in question (probably is easier to use the same input data across all models per batch) and backpropagate to generate the gradients of the inputs with respect to the outputs
1. For each example in the batch:
    1. extract the _unique_ gradients
    1. pick some to mask
    1. create embedder inputs:
        1. all inputs
        1. masked input gradients
    1. outputs:
        1. unmasked gradients
1. feed these embedder inputs into the embedding model
1. loss function is some regression with the correct gradients

We could then see if we're able to train a linear classifier on top of embeddings from this network to classify the different tasks original tasks that produced the models.

The idea is that here we're pretending that these full models are sub-models of a larger model. We could just as easily apply the same method but only _within_ the layers of a larger model. The same principles apply. Here we're instead using our knowledge of the underlying task distribution to test the quality of the the embeddings produced.
