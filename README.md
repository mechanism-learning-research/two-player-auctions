# Auction Learning as a Two-Player Game in JAX

**DISCLAIMER: 
This code is a work in progress and has passed preliminary checks, but might still contain bugs. Thorough testing is yet to come, feedback and questions are very welcome.**

This repository is a JAX/Haiku implementation of the paper ["Auction Learning as a Two-Player Game"](https://openreview.net/forum?id=YHdeAO61l6T) [(extended arXiv version)](https://arxiv.org/pdf/2006.05684.pdf). It uses an architecture inspired by [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) to learn (near-)optimal multi-bidder, multi-item auctions.

The [GAN example from dm-haiku](https://github.com/deepmind/dm-haiku/blob/4ae60fd4fd2da3b2f8f9ad3ec6dfd893745b483b/examples/mnist_gan.ipynb) was used as a starting point.


## Getting started

### Prerequisites
- Install Python 3.7+
- [Install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (optional)
- follow the [instructions for installing JAX with CUDA support](https://github.com/google/jax#pip-installation-gpu-cuda)
- clone this repository and `cd` into it
- run `pip3 install -r requirements.txt`


## Usage

To run the auction experiment with specific parameters:

```
  python algnet.py with num_steps=100 misr_updates=50 misr_reinit_iv=500 misr_reinit_lim=1000 batch_size=100 bidders=5 items=10 net_width=200 net_depth=7 num_test_samples=20
```


## Logging and Artifacts

The project uses the [Sacred](https://github.com/IDSIA/sacred) framework for experiment tracking. 

- Logs and experiment metadata are saved to an SQLite database named `results.db`.
- The state parameters of the last trained model are saved to `tpal_state_params.pkl`.


## Implementation notes
This module is kept simple to make it suitable for use with computational experiment frameworks, or as a component in larger systems.
[Black](https://black.readthedocs.io/en/stable/) is used as a code formatter.
