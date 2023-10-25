# `F = ma` - multi-layer

Now taking `001` and looking at what happens with multiple layers (still maintaining two hidden neurons for ease of visualisation).

## Expectations/hopes

I'm hoping this will help understand what happens when these simple multiplication algorithms that I've found have capacity to grow into neighbouring layers. Maybe that will give some insight into what happens in bigger (more practical) networks.

My expectation is to perhaps see intermediate steps or perhaps multiple rounds of the previous modes to further hone the result. Let's see.

### Where I see this line of enquiry heading (after `001-two-layers`)

As we add more layers, we're adding more possible combinations of operations that will result in similar resultant behaviour/performance and therefore will be similarly favourable in the loss space.

Borrowing a slight statistical physics mindset I almost see this as microstates of a system resulting in (roughly) the same macrostate. If you give your system more degrees of freedom, then it's likely to generate many more energy levels.

### So the question I'm asking myself is: what am I really trying to achieve here?

#### Periodic table of modes

I could continue trying to _understand_ (at a human level) at varying degrees of complexity, what transformations these networks are doing and create some kind of periodic table of different algorithm modes etc. but this seems a little bit stamp collector-y and perhaps isn't telling us anything fundamentally that interesting beyond what is already plainly evident from looking at the transformations anyway.

#### Theoretical mode prediction

I could potentially instead try and create some theoretical framework for predicting these different modes (a little tenuous but perhaps like theoretically calculating electron orbitals to predict chemical behaviour) and see empirically if the trained networks match up. This would certainly be very interesting but also seems very hard given these nasty non-linearities that tend to make life difficult. The end result might be an algorithm much like gradient descent which might not be very revealing... It's almost as if there might not be point characterising the network in any other way but by defining the network itself if that makes sense? Still, it's interesting to think about.

#### Automated structure discovery

Given I can empirically observe these different modes, it might be most interesting to come up with an automated approach for distinguishing them. In this way, such a system could perhaps cluster resulting trained networks into families that could provide some interesting structure. Perhaps by examining how this clustering method operates, I could understand more about the fundamental qualities that separate the different modes more easily. What's more, with such an approach being empirical (as opposed to theoretical), it doesn't tie itself too strongly to a particular architecture/training setup and is therefore more widely applicable to various networks/problems.

This way I could also have a more automated way of looking at training dynamics and phase transitions.

### Where I'll head next

If it isn't already obvious (also from my closing thoughts in `001-f-equals-ma`), I'm feeling quite inclined towards the automated structure discovery. This is less thinking about it from the perspective of: here's a task and let's find what parts of the network are helping towards achieving that task (yet), more looking at mono-task networks and examining what different families of approaches emerge, how they relate to each other and whether there's some other interesting structure that arises (that I likely can't conceive of yet).

First though, I'd like to just close off this thread of increasing the hidden dimension of a single-layer network and whether that makes sense because I'm not sure if I understand how the algorithm changes with increased capacity in this dimension.
