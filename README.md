# Auction Learning as a Two-Player Game in JAX

**DISCLAIMER: 
This code is a work in progress and has passed preliminary checks, but might still contain bugs. Thorough testing is yet to come, feedback and questions are very welcome.**

This repository is a JAX/Haiku implementation of the paper ["Auction Learning as a Two-Player Game"](https://openreview.net/forum?id=YHdeAO61l6T) [(extended arXiv version)](https://arxiv.org/pdf/2006.05684.pdf). It uses an architecture inspired by [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) to learn (near-)optimal multi-bidder, multi-item auctions.

The [GAN example from dm-haiku](https://github.com/deepmind/dm-haiku/blob/4ae60fd4fd2da3b2f8f9ad3ec6dfd893745b483b/examples/mnist_gan.ipynb) was used as a starting point.

## Getting started

### Prerequisites
- Install Python 3.7
- [Install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- follow the [instructions for installing JAX with CUDA support](https://github.com/google/jax#pip-installation-gpu-cuda)
- clone this repository and `cd` into it
- run `pip3 install -r requirements.txt`

### Training models and showing metrics
To training the model and print simple metrics run: `python3 algnet.py`

You can find several settings at the bottom of the file, uncomment the one you want to train.

## Implementation notes
This module is kept simple to make it suitable for use with computational experiment frameworks, or as a component in larger systems.
[Black](https://black.readthedocs.io/en/stable/) is used as a code formatter.
