# F=ma

This is a trivially simple idea just to set up the problem before I move onto more interesting ones.

## The plan

- Define a very small neural network (really doesn't need to be more than a single layer sans activation, but I'll probably try a few things)
- Train it to predict `F` from inputted values `m` and `a` all generated from toy data, maybe I'll add some noise

## Expectations/hopes

Some simple layer will emerge that acts to multiply `m` and `a` together

## Sub experiments

### Experiment 1

This is a simple linear operation (without activation). The final result seems to make sense from a back of the envelope theory perspective in that it represents an optimal solution to the problem.

### Experiment 2

Adding a ReLU activation. The results feel intuitively similar but the optimal parameter values now sit somewhere else. It must be using the ReLU to help it because the loss is now 50% of what it was.

This doesn't feel like such a realistic scenario because you're unlikely to have a ReLU acting on the outputs normally.

### Experiment 3

Now with a single hidden layer (still with a dimensionality of 2).

Breaking down each operation to understand how the vectors are manipulated throughout the "network". Also applying SVD.

When trained to convergence, the results look very similar but the tensor operations (at first glance) look difference. Feel like there _must_ be some equivalence between them so trying to find that.
