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

Looks like ~40% of the outputs are within $\pm 5\%$ orthogonal (i.e. roughly orthogonal). Now need to look into what's up with the rest 🤔

I did a lot of analysis here but only ended up confusing myself. It seemed like the non-orthogonal neurons were contributing less by some measures and more by others. I've decided to simplify my analysis to look at single examples again, just to make sure the metrics I've developed are sensible and I understand them fully.

### `003-multi-mult-ind`

Now focusing on a single example at a time.

This is what I needed to realise that the network wasn't restricted to using the natural basis of the neurons and that the trained models were really just operating in a rotated space the whole time at the `layer.0.act` stage.

Built the intuition that changing the model inputs for one of the tasks simply translated the activations on the hyperplane perpendicular to the other task's output projection - therefore minimising destructive interference. One of those things that seems obvious in retrospect but I'm glad I've learnt it for myself.

Obviously makes sense much more complicated when we're not working in the natural basis but hey _that's the game_.

I've still been thinking about my problem of how one might define the subnetwork that we intend to embed (with the cloze task - see the end of the README for `001-f-equals-ma`). My newest idea: simply see which subnetwork is activated by traversing a certain input space (e.g. changing an input in my particular task) and then picking the parameters according to which ones were involved in the calculation of the final output on a layer by layer basis. I haven't quite worked out the details, but they're starting to form.

I've thought a little about the invariant hyperplane business, and I'm still wondering whether stacked hidden layers will necessarily choose to linearly "access" projections invariant to certain variations in the input space. Surely they could continue to represent this invariance in a "unnatural" basis in the next layer and so on - accessing it in a much more abstract way (to us).

I wonder if you could chase an invariance through successive network layers by [I lost my train of thought] (and whether this would be useful).

### `004-inspect-final-linear`

I intended to get a sense of "how much of the 16D space is being used" by the final linear layer but this seems like a generally difficult one to answer with linear algebra when you're working with rotated bases.

I did some stuff with SVD but I'm not sure how useful it really was.

I suppose intuitively, we'd expect the final linear mapping to use all the information at its disposal (if it has been optimised properly). Perhaps the best way of understanding how much space is being used is actually in the activation space of the penultimate layer i.e. it's pretty clear with PCA across the input dataset how much of the space is being "used".

I've been thinking that it would be good to test whether multi-layer networks trained in this multi-task way (importantly with independent tasks) use orthogonal transformations in their hidden layers. It should be fairly simple(?) to test this by looking at the rank of intermediate weight matrices.

I'm don't have much of a strong hypothesis on whether they _would_ due to these rotated bases i.e. I'd imagine a network could do parallel intermediate processing for two separate tasks without necessarily transforming to something orthogonal in the neuron basis (instead using another basis). Then again, there is something of a symmetry break with the activation functions so 🤷

What I _do_ think though is that it would be _handy_ if there were orthogonal mappings, because that would be some clear flags towards independent operations happening in parallel. If they don't appear then perhaps the way to go is to do some input-wise (as opposed to parameter-wise) analysis - varying the inputs for each task to pull out directions in activation space and then analyse these for independence. I wonder if that could give use a signal to use in order to train an auxiliary model to predict these attributes from the parameters only. Getting a bit too deep in hypotheticals here...

### `005-multi-layer`

Now looking into the aforementioned question of whether multiple hidden layers might choose to represent the separate tasks orthogonally in their activation space.

Hmm looks like there's something here in that there are a few high singular values in the weight matrix with similar values. This suggests orthogonal processing in a subspace which might suggest independent processing? Not sure if my interpretation is correct here though.

The fact that many components have low singular values also might suggest that there is redundancy here. This effect seems to increase with a higher dimensional hidden layer.

### `006-multi-layer-svd-interactive`

I wondered if the SVD decomposed outputs from the middle of the network might be more interpretable in terms of separate tasks but alas it seems not.

Had some thoughts on how perhaps to look for some task separation in the middle layers might be tricky just because the only layer that really matters in terms of task separation is the final one (hence orthogonal subspaces there). Generally quite confused but I suppose we'll work through it.

### `007-single-task-svd`

Just would be nice to understand if there's anything interesting here. My vague hypothesis before trying this out is that we might expect to see fewer of the strongest singular values with a similar value since preserving angles is not as important.

This didn't really play out in a meaningful way. I'm going to change course and try to look at spaces mapped out in activation space by varying the inputs. Perhaps I can see how these are independent in some way?

### `008-multi-task-pca`

I'm going to:

1. vary the input data for one task and do PCA on the middle-layer activations
1. do the same for the other task
1. with the PCA components for each task I can look into whether the activations vary roughly on a plane for an individual task and whether these planes are orthogonal to each other

I think this should offer fairly conclusive evidence whether there is interference between the tasks in the middle layers or not?

Rather than thinking about planes (because the data doesn't sit on simple a plane mate, it's proper manifold business), I just:

- inspected some of the principal components between the two tasks - some nice plots showed that they have some nice structure with some separation and some overlap as expected
- Looks like there is nice orthogonality in the tasks in that I have found a basis in which there is mutually exclusive variance between the tasks

Feels like I'm finally making progress here.

Had some thoughts about what this means for my function detection idea and it seems like I need to work out what to do with this activation space.

### `009-multi-task-pca-dist`

Here I was aiming to run repeats for the past experiment. These were useful in smoothing out the behaviour.

#### Observations

- symmetry emerges between tasks as expected
- previous hypotheses still hold:
  - most variance explained across two components for each task
  - the other task's variance is smoothly distributed between remaining principal components. Would be interesting to understand this distribution with relation to PCA and the geometry of these spaces.
  - there's some interesting structure in the similarity matrices that I don't understand yet but the core facts about the highest variance principal components remain the same

The next experiment is more exciting to me for now.

### `010-combined-vs-separate`

I wanted to test whether the highest-variance principal components extracted from each task separately would also exist in the principal components extracted from the whole dataset (combined tasks). This would be very handy because it would remove the need to have prior knowledge about what the tasks are and how to vary them independently (which would become very hard if we start to think about abstract notions of internal tasks).

Intuitively, I'd hope that there is some relationship between these two (or three) sets of principal components.

Turns out there was, and it was quite nice 👍 (it's also simpler to be just considering one set of PCs again 😅 and not worrying about directionality).

There seemed to be a nice independence between the task association for each component. In fact, interestingly it looks like on average ~80% of the variance is explained by the dominant task for each component - across essentially all the components 🤔. We also see that for this task, essentially all the important variance is contained within the first four components.

This is nice. But we're left with the issue of how to cluster the components into their separate tasks. Some ideas here:

1. Look at covariance between the components within the same layer - but I'm not sure if this would work out properly.
1. Look at covariance from PCs in neighouring layers and potentially build out some graph structure from the dependencies. _This_ seems like a fun idea.

I mean _surely_ someone has tried these before?

I think I'm going to look at 2. for the next experiment. Might be a nice point to move on to a new master experiment.

## Experiment summary

My aim in this experiment was to see whether I could understand how the MLP would treat two independent tasks. I think I was generally hoping to see some orthogonal processing going on.

I started looking at things in the neuron basis and after much wandering around, finally built my intuition that the MLP rarely seems to ever work in the neuron basis. In fact, it feels like it's _beneficial_ to work in a basis misaligned with the neurons.

I found the first evidence of task orthogonality in the final linear mapping to the outputs.

I then focused on whether there existed orthogonal processing in the intermediate layers of an MLP and spent some time looking at the parameter space without much luck. However, once I switched to the activation space, things became much clearer and I was able to find evidence of this.

I built out some PCA tooling to successfully (I think) find the relevant task components for the two separate tasks in my hidden activations and show that they're more or less independent between the tasks.
