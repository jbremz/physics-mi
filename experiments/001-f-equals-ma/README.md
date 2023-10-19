# F=ma

This is a trivially simple idea just to set up the problem before I move onto more interesting ones.

I suppose, in retrospect, this experiment is investigating how multiplication between two continuous values is typically (optimally?) carried out in MLPs.

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

Looks like I could split the resulting trained models into 3 rough modes. I looked at the most promising one in more detail and came up with some early hypotheses on how it is carrying out multiplication.

Introducing seeds too 😅

### Experiment 4

Same setup as Experiment 3 but now finding a better (cleaner) example of the 3rd mode of resulting model with more insight on what is happening between the ReLU and final linear layers.

### Experiment 5

Following on from Experiment 4, I've changed the data input distribution (to be uniform in the _input_) to show that it alters the skew of the transformed unit square at the `preacts` stage. This makes intuitive sense as it focuses the model's performance more on the smaller values.
