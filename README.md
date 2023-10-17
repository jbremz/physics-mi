# Mechanistic Interpretability with physics data

I have this idea to look inside models that are trained to model physics data (raw measured physical quantities) to examine how they learn relationships and whether there are any emergent physical models that I can discern.

Maybe this is a stupid idea but I'm keen to play around (without having done _too_ much of a literature review) as that'll probably provide me with the most intuitive insight.

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
