# Mechanistic Interpretability with physics data

I have this idea to look inside neural networks that are trained to model physics data (raw measured physical quantities) to examine how they learn relationships and whether there are any emergent physical models that I can discern.

This is an idea that already exists in the field of symbolic-regression. However, in SR a typical approach would be to train auxiliary models which have been constrained to certain symbolic forms so that they accurately match the input/output behaviour of a less constrained model. Here I am trying not to apply too many constraints on the central model (beyond potentially regularisation) and to instead use mechanistic interpretability techniques in order to try and understand the internal mechanics of that central model itself.

As you will quickly see, this lofty goal first requires detailed understanding of how low-level mathematical operations are carried out within neural networks.

**NOTE: this repo currently represents research in progress so is unlikely to be _completely_ tidy, such is the research process.**

## Structure

- `experiments` - where I will lay out my individual experiments (primarly in jupyter notebooks)
- `src` - where I will maintain a package with useful tooling for across the experiments

## Installation

In order to setup the environment and install the research tools:

```bash
conda env create -f env.yml
conda activate myenv
pip install -e .
```
