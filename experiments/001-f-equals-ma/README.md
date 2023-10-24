# F=ma

This is a trivially simple idea just to set up the problem before I move onto more interesting ones.

I suppose, in retrospect, this experiment is investigating how multiplication between two continuous values is typically (optimally?) carried out in MLPs.

## The plan

- Define a very small neural network (really doesn't need to be more than a single layer sans activation, but I'll probably try a few things)
- Train it to predict `F` from inputted values `m` and `a` all generated from toy data, maybe I'll add some noise

## Expectations/hopes

Some simple layer will emerge that acts to multiply `m` and `a` together

## General Qs on my mind

- I can "discover" various multiplication algorithms in a 2-d hidden layer, but how will this translate to higher dimensions? Who's to say that the most efficient algorithm in 2-d generalises to higher dimensions? Who's to say that the increased algorithmic capacity afforded by increasing the hidden layer dimension doesn't lead to more complex, more accurate ways of modelling this operation (akin to higher energy configurations being unlocked)?
- I could find some kind of probe for determining whether a certain algorithm is being used in a network, but would this work well once the network needs to do multiple operations at the same time? Does superposition occur with algorithms modeling continuous operations and if so, what does it look like? Would methods like Anthropic's sparse autoencoders (that were originally intended for categorical "features") even make sense in the continuous space?

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

### Experiment 6

I hand-crafted my own algorithm to see where what really mattered and if I really understood what was going on. Didn't do too bad a job although it wasn't optimal.

### Experiment 7

I took the hand-crafted model from Experiment 6 and optimised it further with backprop. The results were interesting in reintroducing the skew but not really changing the y-displacement at `preacts` stage.

### Experiment 8

I played around with the initialisation of the bias term in the handcrafted network from Experiment 6. Turns out it never really gets optimised too much when I then finetune. I think this is more a result of the training procedure and getting stuck in local minima easily (much easier when starting from an already somewhat-optimised network).

### Experiment 9

Here I return to simply training the network from scratch with various random initialisations of data and network just to see if my "discovered" algorithms (modes 1, 2 and 3) make sense.

Realise there's a mode 2.5 that I hadn't considered before and otherwise understood a few more things about modes 2 and 3 (see plots in `plots/`).

### Experiment 10

Now I'd like to see if I could develop some kind of "probe" for the existence of these algorithms. This might also allow me to examine the training dynamics e.g. are there phase transitions between these algorithms?

Ideally, I'd like this to lead me towards some kind of probe I could use in higher dimensions (where it's trickier to do visualisations).

**NOTE:** maybe I should stop using the word "probe" as it seems to be used to describe something slightly different in the literature (although the general aim is similar) see [here](https://arxiv.org/pdf/2102.12452.pdf). Perhaps scan or algorithm scan might be a better term?

In the end I manage to hack together a scanning function that has 100% AUC against randomly initialised networks. Doesn't feel very neat though and I realise there are important questions to answer about how one might apply this to larger networks. As a result I think I might not continue pursuing this _just_ now (even though I think it still has worth).

## Closing experiment remarks

I suppose in retrospect this umbrella-experiment (with 10 sub experiments) was looking at single-layer networks with two hidden dimensions and how they perform multiplication. I'm pretty confident I've answered that and found more interesting detail than I expected...

Now feels like a good time to start a fresh high-level experiment as conceptually I would like to extend beyond this scenario into:

1. more layers
2. more hidden dimensions

I'm going to start with 1. because I _feel_ like for this problem at least it should be simpler to understand as I have already built quite expressive tooling for visualising activations/layer outputs in 2 dimensions.

**Just to write it down _somewhere_:** I've been having lots more thoughts on how one could search for particular internal circuits/algorithms but in a more unsupervised way i.e. _without_ knowing what we're initially looking for. It seems to me that one would need to train some meaningful representation given the input of certain parameters/parts of a network.

How could you do this? I wonder perhaps whether you could do your standard cloze task but with a regression on the obscured parameter values? The network tries to "fill-in" the missing parameter values, the intuition being that it can use the context to understand _what a certain group of parameters are trying to achieve_ and therefore use this to help it predict the neighbouring parameters.

Regardless of the specifics of the training process, I'd hope then that you may have a general feature extractor which you could use to characterise different sections of the network. One could e.g. perform clustering on the resulting embeddings to identify functionally equivalent parts of the network?

Potential issues/questions I see here:

1. To train something like this you might need a big network to train on (as the subject network) and the investigator network would need to be an appropriate size not to overfit on it
2. Would there be transfer between different networks e.g. could you train it on one subject and then extract meaningful features on a separate network? I suppose this would depend on architecture etc. Maybe it would still be useful when trained and evaluated on the same network?
3. There are questions around input: what would the input look like? How would you deal with the potential of different hierarchies of algorithm size (i.e. how do you define the unit of interest)? Maybe there could be a variety of sizes inputted with padding. How would you tell the network which parameters correspond to which types of operation? Would you have a sliding window at inference?
4. Potentially this training task is just too difficult to achieve to get nice clean representations?

Would be cool to think of other unsupervised methods for generating parameter representations (you could _almost_ call these meta-representations but not quite).
