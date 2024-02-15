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

### Results

Maybe there's _some_ structure in this space? I'm not sure if that's trivially true though anyway because each function has a different shape and we're outputting the shape 🤷 I'm also aware that my model has overfit and generally isn't performing so well. I think for these we need to train a nice well-performing model before we see any clean results.

I'm going to roll back a bit in the next notebook and try and simpler straight up classification of the functions from their gradients. This should be easier and hopefully at least tell me that enough information is there. Then the task goes back to whether we can come up with an unsupervised method for extracting embeddings (because in real-world applications we cannot train in a supervised way).

## `02-func-classifier`

As I just mentioned, it's simpler to take these models and train a classifier just so I'm not making _too_ much of a leap at once. I'm starting to push the limits of what I can sensibly train on this MBP 😭

### Results

Ok this was straightforward.

To be honest, this seems extremely trivial. I can _see_ that the gradients are distinct (and therefore classifiable) between each function, so being able to train a classifier on it is no surprise. _Now_ it's just a case of training something _unsupervised_ to pull out the same information 🤔.

One potential issue I forsee is what happens when your input space becomes loads larger. Obviously 1D is fine because we can sample densely. I'd imagine for high dimensional input spaces we'd just need bigger models...

I've been thinking a little more about whether what I'm doing is _generally_ really trivial. For example, is this predict/embed-the-function-from-the-gradients thing the same as basically predicting the raw outputs from the raw inputs i.e. are we not essentially just training a model to replicate another model? Well, firstly, not quite because the gradients are a more _functional_ view of the element we are examining (e.g. MLP) than the pure input/output space - given dense enough sampling (so that we have degeneracy in the gradient space) it is telling us what is this element _doing_ to the input in this _general region_ of the input space, in my head it provides more structure. What's more, even if our trained inspector models are somewhat replicating the element under examination, they produce an output that is much more useful to us (hopefully) i.e. one that contains information about the _element_ not the inputs.

## `03-embedder-again`

Now I know I can definitely classify the different functions well from their gradients, I'm going to continue with my previous idea of doing the same thing but in an unsupervised way.

### Results

Brought it back down the cloze task for only two functions and we now have nice clustering :)

## `04-bottleneck`

Now that I've got an unsupervised model embedder that creates a meaningful representational space (albeit with only two underlying functions to embed) I'm going to try and crank it up a bit (to embed more functions). One idea I've had is to create a bottleneck layer that will hopefully force the network to squeeze the models into their separate classes.

### Results

- Works nicely for 5 functions (and a bottleneck dimension of 4) with a linear probe accuracy of ~90% for separating into different functions
- In order to get good representations there seems to be a delicate balance between:
    - number of distinct functions to represent
    - the information contained in each function (distinguishability / number of high freq. components)
    - capacity of base models
    - how much masking to use in the cloze task
    - the bottleneck dimension
- This I suppose is to be expected with representation learning
- when I increase to 10 different functions, there is still separation but performance drops off much more (linear probe accuracy of ~40%) - will need to think about how I could scale this method well

### Thoughts

Taking a step back again, we have achieved:
- a process from which we can cluster together (to some degree of accuracy - which I'm sure can be improved) different neural networks in terms of what function they are modelling using only access to the model and a typical data distribution for that model

This maybe doesn't seem very useful in this case because we're applying it to full models where we have prior knowledge of the function they're modelling (because we've trained them e2e). In order for this method to be useful we need to answer:
- can we apply this internally in a network to study internal structure?
    - does a higher dimensional input space cause lots of problems with scaling (it really might)
    - what units of the network should we study? Just layers? Groups of layers? Can we train embedders that are scale invariant?
- if so, how can we verify that the structure extracted is indeed useful to us? How do we make sense of these embeddings?

My hope is that if we're be able to pull out some similarity scores for elements within a network, then we might be able to piece them together to move towards _classifying_ what they're doing.

### Next steps

I kind of wonder if we're getting towards an MNIST example 🤔 concrete steps to achieve that:
1. Train an MLP on MNIST (as small as possible with good accuracy)
1. Train my embedder with various different input and output nodes - maybe I could contrain this more at first?
1. see what happens with the embeddings?