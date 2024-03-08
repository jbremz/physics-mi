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
- Now I think we can go onto the embedding

## `03-iris-mlp-embed`

Ok so, I've been thinking, and one nice experiment I thought I could potentially do is to train my multi-task networks again with their identical but independent tasks so I would _know_ that there should be similar functional behaviour across each "branch". How I'm seeing it:
- stack up Iris data randomly so that we have 6 inputs to our MLP (two sets of 3 inputs each) and 6 outputs. Each input/output pair is randomly sampled from the Iris set.
- applying my gradient field embedding method to various computational units of the network

To start with, I think it's a good idea to constrain. I think what I'll do is focus on modelling a single layer unit i.e. a component and it's first generation "children" in the preceding layer. I _hope_ to find similarities across child units of each task since they are modelling the same data.

### Results

Struggling to get any intelligible embedding space here. I think I may have over-reached.

I think what I'll do next is return to the problem I examined in `008-net-embed/05-bottleneck-satic-grid.ipynb` and increase the input space dimensionality gradually. If I can get it working well with 64D data and toy data then maybe I'll be in a better position. If it still doesn't work, then it'd be a case of understanding at what point it breaks down.