# Layer correlations

I made some good progress in `004-multi-task`, understanding orthogonal task-treatment in multi-task networks and successfully identified the different components related to different tasks. Ended up being quite simple really...

_Now_ I've had an idea to identify sub-networks corresponding to different tasks, _without_ prior knowledge of the different tasks. It's probably inaccurate to call them sub-networks, perhaps subspaces is a better term because we won't be working in the neuron basis but instead in the principal component basis (defined by the variance in the input data distribution).

## General thoughts on my mind

- I think I'm still more drawn towards parameter spaces but I'm hoping that understanding the activation spaces might help as a way into the parameter space problem.
- [This circuits thread](https://distill.pub/2020/circuits/branch-specialization/) on branch specialisation looks to be quite related to this work (albeit with CNNs). They use SVD (as I was before) on the weight matrix to find some evidence of branch specialisation (maybe what I call task independence or independent/orthogonal task treatment). Might be nice to tie these two views up.

## Experiments

### `001-subspace-graph`

The idea is very simple and goes:

1. Keep same data and network as `004-multi-task` - potentially add more depth to make our graph more interesting if we need to
1. Get all the intermediate activations for all the layers when passing a full dataset through the trained model (still carefully labelled so we can keep track of which task is which for evaluation)
1. Apply PCA on the activations layer by layer
1. Compute covariances between the PC activations between separate layers
1. Somehow use these covariances to create something of a computational graph of subspaces that correspond to different tasks (I'm shaky on the details for this part)
1. Compare the labels produced from the previous step with ground truth labels that we can produce from our prior knowledge of the tasks involved

### `002-check-rand-dist`

Holiday me realised that I should check that the results I produced in `004-multi-task/010-combined-vs-separate-components` also stand when I extract components from a more typical input distribution where the tasks vary simultaneously (not this nice dataset I'd created where they vary independently).

Looks like the same results hold 👍 this is good because it means we have a good chance of extracting the orthogonal task components from a validation-set input distribution (which we will always have).

It did raise the question of whether the ~20% task interference that _does_ exist originates from:

- inherent non-orthogonality in the processing of the network, or
- limitations in the use of PCA to find the orthogonal components

Had some ideas about how one could see if the parameters contain information for these orthogonal task components using a regression modelling task. This would be cool.

Also had some ideas about task capacity in MLP layers.

### `003-subspace-graph-revisit`

After the confirmation of the last experiment, I decided to change my work flow so that I now used the PCA components from the validation set throughout.

Here I followed the same path as in `001` but stopped at the correlation matrix because I'd realised that this kind of analysis was unlikely to produce the clean results that I wanted. Of course, the next layer is going to use various weighted _combinations_ of the previous layer outputs to map onto the principal components of the next layer, and equally, it might be finding spaces orthogonal to hyperplanes traversed by the opposite task's components so as to remain independent to them. This would be very hard to read off a correlation matrix I feel.

This feels like a similar trap to the one I first fell into with `004-multi-task`.

With this understandig, one _could_ I suppose try to find groups of principal components in the previous layer that correlate well or produce close to zero correlation with the next layer PC outputs to divide the tasks up, but this seems to be a combinatorially hard problem and one that might not scale very well.

I had an idea: why not use PLSR (partial least-squares regression) to find the components in the layer0 output space that correlate the most with the layer1 output space. These won't necessarily be the principal components from layer0, in fact, we'd expect them to be linear combinations of the principal components from layer0. From finding the overlap between these PLSR components in layer0 space and the PCs from layer0 space, then we can hopefully start grouping with a graph structure.

I'm really unsure whether this will work but let's see and maybe I'll learn something.

### `004-inter-layer-plsr`

Ok so, here's the plan:

1. get multi-layer activations for validation set
1. get multi-layer activations for task datasets
1. calculate PCs from valid_acts for both layers
1. use the PCs from layer1 as our dependent variable and the activation space of layer0 as our multivariate input independent variable and perform PLSR
1. take the resulting layer0 PLSR components and see how they overlap with PC components from layer0
1. maybe we can produce some groupings of the layer0 PCs from this process

Seems quite tenuous at this stage but hopefully we'll learn something.

The results look good. There is fairly clear task-separation between the input layer PLSR components. It seems from this that one could build up something of a computation graph between components for subsequent layers. I put down an idea to then display this graphically by creating two sets of components per layer:

- one is the output space for that layer - formed from doing PLSR with that layer and layer before
- the next would be the input space for the next layer - formed from doing PLSR with this layer's outputs and the outputs of the next layer

This could look nice, are there would be intra-layer weights (between the two different sets of components) too, which would represent the mapping from the raw features in the activations (those which explain the most variance in the data) and the components which are important for the next layer (which could of course be mixtures of the other features). This could show how features are combined in a nice way e.g. I'd imagine for this multiplication task that at some point before the last layer, we'd see mixing between the magnitude features for the pairs of numbers involved in each multiplication but importantly _not_ between the independent multiplications.

**But I have a more fun idea**: the problem with this PLSR business is that it uses a linear assumption for what we're modelling which is _almost_ correct except for our non-linearity. It evidently works roughly speaking at this single linear layer level but if we were modelling a not-so-linear layer then I'd imagine it might not so well.

Really, it seems to me that back propagation is closer to what we're looking for. We could set up a system whereby we create a further (temporary) linear layer that simply linearly accesses the PCA components we've extracted, _then_ we do backpropagation on each of the output logits from this layer to the previous layer activations in order to find the components there that maximally activate each output logit.

Questions/issues:

- I suppose we'd need to run this across a number examples and average the resulting vectors - roughly as we have been with PLSR using the validation set
- maybe that's it?

This would hopefully model the non-linearities more faithfully (completely faithfully?).

### `005-inter-layer-backprop`

I've since read about a few people trying similar things but for different purposes (e.g. this recent post [_Case Studies in Reverse-Engineering Sparse Autoencoder Features by Using MLP Linearization_](https://www.lesswrong.com/posts/93nKtsDL6YY5fRbQv/case-studies-in-reverse-engineering-sparse-autoencoder)) so I feel like this is a good idea. Lots of people sniffing up this particular tree.

I created a very simple "scaffold model" which essentially loads an internal layer of the network (e.g. layer1) and the previous layer's (layer0) activations (i.e. inputs to layer1) and then computes the gradient of a single PCA component in layer1's activations with respect to each dataset example.

Interesting things I've noticed:

- I then ran PCA on these input activation gradients to (if my intuition is correct) find the components in the input activation space that produce the largest change in the PCA component I'm studying. It seems that most of the variance tends to be explained in the first four or so components although I need to be careful to get my head around which ones are actually making the most difference by joining up high variance components in this layer0 input space to high variance components in layer1 output space. I think it's high time I created a plot to stop me going crazy.
- One thing I haven't been able to explain is that when I plot a dimensionality reduction of these activation gradients, I find that 18 very neat clusters form 🤔 Perhaps there's a neat explanation for this but what it's telling me right now is that within in the 2000 examples I have, there are 18 distinct directions in the activation space that are affecting the layer1 PC being studied. Some of these clusters seem much bigger than others, I'm going to try plotting the principal components on this same plot in order to see if there are any interesting relations with the clusters.

You can read in the notebook for more detailed thoughts (as always) but I have a line of enquiry I'm going to follow next which will hopefully help put to bed this question of whether there is orthogonal task processing happening internally (as we suspect). If I can answer this question soundly then I can move onto the question of _how_ to pull apart the internal representations into separate tasks in an unsupervised way.

### `006-backprop-from-outputs`

#### Proposal

- backprop from task outputs to each layer to generate activation space gradients
- apply PCA to these gradients (as we have done above)
- we will then have two sets of components, one for each task for every set of activations (I suppose all the way back to the individual inputs)
- compare these two sets of components and test for orthogonality (at least in the highest variance components for each task).
