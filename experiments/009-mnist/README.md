# MNIST

Ok potentially a little premature, but I'd at least like to try this because it could be informative and at least keep me focused on the practical application of this idea.

## General thoughts

- Need to remember to not use the cloze task dataset when producing my embeddings. This will probably make the embeddings much more noisy because you're masking half the signal!

## `01-mnist-mlp`

I'll try and train the simplest MLP I can to get decent accuracy on MNIST. Could totally copy a tutorial but let's see what I come up with myself hehe.

I don't have a _really_ good idea of what I'm going to do then until I've felt it out.

Will probably look at the task independence stuff too (à la `007-comp-graph/005-soft-mixing`) which will be interesting because I'm not sure the extent to which task mixing appears in _real_ (not just toy) datasets.

## Results

- Ok running the task independence analysis proved a little too slow (on my laptop) - I'm sure there's some pretty nasty exponential scaling going on.
- Would need to think about maybe breaking this down into studying sub trees of the gradient tree
- for now probably go to an even smaller dataset

## `01-iris-mlp-task-sep`

Same as before but with this smaller dataset. I decided just to focus on the task independence part for this because the next part seems erm long 🙈

### Results

- this worked out nice
- interesting to see the differences between this classification task vs the toy regression tasks I've been using. General point is that you start to see inter-class gradients negatively interfering from quite early on - this makes sense as the presence of one class is negative evidence for the presence of another :) in this way, the tasks are definitely "mixed"
- There is also mixing between the first two classes suggesting that there are some parts of the input space in the 2nd class for which similar components to 1st class will be used to increase the 2nd class' probability.
- Now I think we can go onto the next 