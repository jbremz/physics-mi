# MNIST

Ok potentially a little premature, but I'd at least like to try this because it could be informative and at least keep me focused on the practical application of this idea.

# `01-mnist-mlp`

I'll try and train the simplest MLP I can to get decent accuracy on MNIST. Could totally copy a tutorial but let's see what I come up with myself hehe.

I don't have a _really_ good idea of what I'm going to do then until I've felt it out.

Will probably look at the task independence stuff too (à la `007-comp-graph/005-soft-mixing`) which will be interesting because I'm not sure the extent to which task mixing appears in _real_ (not just toy) datasets.

## Results

- Ok running the task independence analysis proved a little too slow (on my laptop) - I'm sure there's some pretty nasty exponential scaling going on.
- Would need to think about maybe breaking this down into studying sub trees of the gradient tree
- for now probably go to an even smaller dataset

## `01-iris-mlp`

Same as before but with this smaller dataset.