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
