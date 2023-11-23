# Layer correlations

I made some good progress in `004-multi-task`, understanding orthogonal task-treatment in multi-task networks and successfully identified the different components related to different tasks. Ended up being quite simple really...

_Now_ I've had an idea to identify sub-networks corresponding to different tasks, _without_ prior knowledge of the different tasks. It's probably inaccurate to call them sub-networks, perhaps subspaces is a better term because we won't be working in the neuron basis but instead in the principal component basis (defined by the variance in the input data distribution).

## General thoughts on my mind

- I think I'm still more drawn towards parameter spaces but I'm hoping that understanding the activation spaces might help as a way into the parameter space problem.

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
