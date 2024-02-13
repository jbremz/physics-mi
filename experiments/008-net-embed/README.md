# Network Embedding

Here I'm going to try out an idea I had to train a model that could create a "network embedding", that is, an embedding that describes the "functionality" of a network i.e. what it does. This is hopfeully a step towards creating a system that can automatically produce a structural analysis of "components" (e.g. subnets) of larger networks.

To repost some thoughts I laid out in `007-comp-graph`:

In the back of my head I've still been wondering how I might train an unsupervised system to embed different parts of the network. Here's my latest idea (based on the thoughts above):
- provide:
    - intermediate activations (a sampling of the space)
    - the gradients produced between these intermediate activations and some component later on in the layer (how far later depends on the granularity of your study)
- train it with a cloze task. Given _all_ of the inputs and only some of the gradients the model needs to fill in the missing gradients, and in doing so hopefully understand the underlying function it represents. This amounts to predicting piecewise-linear sections of a larger function.
- can extract an embedding which would hopefully represent some information about the underlying function being modelled by that part of the network (on that data)

How I might test the usefulness of this method:
1. Train a variety of different (single) tasks - probably different mathematical formulae for now. Multiple trains for each task so I build up my "dataset" of different models e.g. 10 tasks x 50 models/task = 500 models. 
1. Partition my dataset traditionally so that each task appears in both the training and validation sets but _importantly_ with different seeded trained models e.g. 10 models per task are in the validation set
1. Generate new input data each batch and run it through the models in question (probably is easier to use the same input data across all models per batch) and backpropagate to generate the gradients of the inputs with respect to the outputs
1. For each example in the batch:
    1. extract the _unique_ gradients
    1. pick some to mask
    1. create embedder inputs:
        1. all inputs
        1. masked input gradients
    1. outputs:
        1. unmasked gradients
1. feed these embedder inputs into the embedding model
1. loss function is some regression with the correct gradients

We could then see if we're able to train a linear classifier on top of embeddings from this network to classify the different tasks original tasks that produced the models.

The idea is that here we're pretending that these full models are sub-models of a larger model. We could just as easily apply the same method but only _within_ the layers of a larger model. The same principles apply. Here we're instead using our knowledge of the underlying task distribution to test the quality of the the embeddings produced.

## `01-1D-functions`

To really make this problem as simple as possible to begin with I'm going to randomly generate a "family" of **1D polynomials** as my functions to model, maybe 3rd or 4th order to get some weird shapes in there. This will hopefully mean that my models can remain quite small as the input to my network embedder will be simply a load of randomly sampled 1D inputs along with their 1D gradients which will mean I'll have a $2N$ dimensional vector for $N$ samples. Otherwise, I think I shall follow the procedure that I have outlined above.

My priorities will be (in order):
1. make sure my polynomials look reasonable
1. make sure I can successfully train a small MLP to model the polynomials to a reasonable degree of accuracy
1. make sure the unique gradients I observed in 2D holds in 1D also (don't see why not?)
1. maybe make some pretty plots of the gradients to also observe the line-sections more clearly (like I did with the vector fields) and to compare them with their theoretical gradients
1. train the embedder (should I call it feature extractor?) model on the cloze task, removing certain unique gradients for it to predict (need to come up with a suitable 1D mask value) - make sure this trains ok otherwise we probably don't have much hope of seeing structure in the embedding space
1. examine the structure of the embedding space with respect to the different underlying polynomials that produced the data points. Can we cluster them? Can we train a linear classifier on top of them?

### Thoughts

- need to get enough variety in the polynomials, they tend to look the same with a naïve sampling strategy