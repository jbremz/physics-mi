# Multi-task networks

The previous experiments have focused on a single (very simple) mathematical operation and how it is achieved in higher dimensional single hidden layer MLPs as well as 2D two-layer MLPs.

This gave me quite a good understanding of what kinds of algorithms the network learns to achieve this single task.

In reality, if we're still thinking about modelling physical systems (as is still the high-level aim of this research), there will be a combination of various different mathematical operations required to be computed in parallel and combined in various ways. Now I think we can move to focus on this situation and whether we can use what we've already learnt to understand parallel computation in MLPs.

One of my main concerns here has been with the potential of superposition occurring and the difficulties in disentangling algorithms in superposition. _However_ having thought about it a little more, I'm _cautiously_ optimistic that superposition might not occur in this situation.

The reason being, is that in the original Toy models of superposition paper, they demonstrated that it is sparsity (in their case with categorical/binary features) that allows superposition to occur. When they decreased the sparsity (i.e. increased the density) they found that the superposition disappeared.

In a continuous-value regression problem, the inputs are by definition dense i.e. _the features are always there_ (without rigourously defining what I mean by a "feature" in the continous sense). In this way, I find it hard to see how there could be parallel computation of values using the same neurons without there always being destructive interference (for uncorrelated input values).

So the hypothesis is that the redundant neurons that I observed in `003-f-equals-ma-multi-dim` could be employed to model other tasks. I suppose the interesting situation is when there do not exist enough redundant neurons to "fit" an extra task.

## Problem setup

The setup I have in mind:

1. Since I understand multiplication ok now, let's continue with that but do two parallel multiplications of two pairs of numbers - so our input will now be four numbers.
1. The output will be two numbers i.e. the result of each of the two multiplications

I can do similar plots to the ones I was using in `003.006` but for each output independently to see whether there is separation in the "activated" neurons for each task.

As a note, what makes this kind of problem much easier at the moment is that I can cover the input space essentially entirely (as the inputs are just 2D bounded values). This means I can be much more confident in my analysis as opposed to having to pick my input samples carefully.

## General things on my mind

- Continuing my thoughts from the end of `001-f-equals-ma`, I've been thinking more about the issue of choosing the right parameters to feed into a cloze task e.g. do you feed in a whole layer (but then lose granularity) OR you could use some kind of neuron importance metric derived from the above problem setup (i.e. which hidden layer neurons are "activated" for each output neuron - maybe conceptually similar to ACDC) to define a subset to operate over. This feels to me to be a potentially neat way to automatically partition the network into functionally separate entities.
- One idea I have: train two separate single-task models, pruning them by 50%, and then combining them into a single _multi-task_ model. Then I can train this model further to understand whether, from this (potentially local) optimum, there is a mixing of the task functionality (and if this is favourable for performance). My intuition is that if this happens then it shouldn't result in a lower loss. Perhaps introducing regularisation would force the model to share more functionality between the tasks though.

## Experiments

### `001-multi-mult`

I set up the problem with parallel multiplications and found some interesting results involving _some_ neurons that seemed to be mono-functional (i.e. only serve the result of one of the tasks) and some that seem to contribute towards both. Their distribution looked interesting on some of the training runs I tried but I needed more data to really understand it. Hence...

### `002-multi-mult-ind-dist`

I wrote a script `train.py` to train and evaluate 100 models.

In the notebook I go into analysing the results.

Looks like ~60% of the outputs are within $\pm 5\%$ orthogonal (i.e. roughly orthogonal). Now need to look into what's up with the rest 🤔
