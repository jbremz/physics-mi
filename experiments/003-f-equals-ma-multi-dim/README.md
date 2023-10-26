# `F=ma` - increasing the hidden dimensions > 2

We've understood the `001-f-equals-ma` modes fairly well with 2 hidden dimensions, as well as having some idea of where things are headed when we add more depth (hidden layers) in `002-f-equals-ma-multi-layer`.

Now, I'd like to understand what happens as we travel from a setup as described in `001-f-equals-ma` towards a single-layer network with more hidden dimensions (i.e. > 2).

## Potential struggles

Really I see these to be centred around: visualising (and by extension understanding) > 2 dimensions.

I might have to say goodbye to my beloved 2D unit-square transformations 😭👋

Some ideas I have here:

- continue to plot 2D unit-square transformations but simply choosing one pair of dimensions at a time
- some kind of dimensionality reduction but never that happy about this

## Expectations/hopes

In my head, I'm imagining that we're going to head towards a similar behaviour of aligning to axes and applying ReLU partial "projections" followed by the linear projections that use the extent of different points in the positive side of the axes to further separate the lower from the higher products.

The difference here I suppose is the increase in information one can store in each component. For example, in 3D, the extent in the 1st two components could be much the same as in the 2D case, but the third component could store finer discriminative information.

I'm wondering now too whether we'll continue to see the optimal solution containing a ReLU projection on only one of the axes or whether that will extend to two axes? 🤔 Perhaps the mode 2 of the 2D case will in fact be more optimal in this case. Let's see.

## General thoughts/Qs

- It seems that throughout these experiments, my training often gets stuck in local minima. I put this down to a lack of regularisation i.e. I have used _none at all_ (and the choice of gradient descent as opposed to _stochastic_ gradient descent probably hasn't aided here, as pointed out by someone helpful). Maybe this is something I should think about if I'd like to arrive at more optimal solutions?

## Experiments

### `001-3D`

Describe one of the first resulting networks I train.

Realise the 3-4 modes I had identified in `001-f-equals-ma` were only one part of the picture and identify 15 different combinations of 2D quadrants (some of which are symmetrical) which the unit square input can be transformed to sit over after `layers.0.linear.bias`.

### `002-3D-more-modes`

Find more optimal training runs and intuitively understand the result. It played out roughly as I had laid out in the Expectations/hopes section of this README.

Now starting to think about the more general trend in higher dimensions still. How do these findings generalise?

I can either think my way to the answer or look at it empirically 🤔

I suppose pursuing the former (followed by confirming with the latter), I'd imagine this behaviour to generalise fairly straightforwardly in higher dimensions? That is, **dividing the task of projecting different parts of the unit square by using different components** - I suppose this is not unlike the piecewise linear view of ReLU-based neural networks (where each node deals with different subspaces of the input in a linear way).

### `003-16D`

Now really upping the dimensionality. Hopefully this is a fairly general example of a "higher" dimensional hidden layer.

### `004-4D`

Developed a better plotting function for visualising the effect of different neurons on the final result across the unit square.

### `005-4D-more-modes`

Start to observe more structure amongst the roles of different neurons and that there are often a handful of neurons that do the brunt of the work and the rest are for finer adjustments.

### `006-16D-revisited`

Now with better graphing functions I can revisit the higher dimensional case. Found that there are (understandably) a lot of solutions but also a lot of redundancy.

#### Some questions that arise from this

1. are there any more interesting underlying dynamics going on here? I think I can currently intuitively understand that the random initialisations of the weights (and data) almost predisposes some nodes to be destined for certain roles more than others (especially with full gradient descent). There are many different favourable configurations of linear transformations (coupled with the static ReLUs) that can result in good performance once you increase the number of hidden dimensions and we're seeing them here. **There might not be much more sense to this than that** - at least that's what I'm feeling right now.
1. I suppose one could look at scaling laws of the "active" vs "redundant" nodes (should I be calling them neurons?) with respect to various conditions
1. One question this _does_ raise though, is that since we evidently have redundancy in this network, what happens when we increase the complexity of the task to be learnt? This could take many forms e.g. multiplying > 2 numbers, doing multiple different operations at once etc. Do these then end up "filling up" the other neurons and operating in parallel or does the network find more sophisticated ways of distributing this task in parallel across neurons (some kind of continuous functional polysemanticity 👀). This could be interesting to understand.

On the final point, I think this kind of idea very much relies on identifying the relevant neurons for a particular portion of the task. I suppose one could explore this by having two outputs (one for each task) and examining whether we see a sharing of the computational load across the neurons or whether they remain functionally partitioned. I think this could work and could be straightforward to try out 🤔

Could be some interesting effects of increasing functional redundancy/separation with increasing hidden dimension a la Toy models of superposition paper... (and inversely: increasing superposition at smaller hidden dimensions? Whatever this might mean in the continuous space?)

My intuition would tell me that the two output case would generalise then to intermediate calculations that are used by later layers in deeper networks but this would need to be checked.

I'd hope that this would at least help on the journey towards functional classification of different groups of neurons.
