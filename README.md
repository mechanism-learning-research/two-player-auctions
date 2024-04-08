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
python algnet.py with num_steps=100 misr_updates=50 misr_reinit_iv=500 misr_reinit_lim=1000 batch_size=100 bidders=3 items=10 hidden_width=50 n_hidden=3 num_test_samples=20 attack_mode=None
```

You can also run the experiment with parameters given by a configuration file. You can find some example configs in [baseline_configs](https://github.com/mechanism-learning-research/two-player-auctions/tree/main/baseline_configs).
```
# run experiment with parameters from config file
python algnet.py with baseline_configs/config_2x2.json
```
```
# run experiment with parameters from config file, overriding the number of hidden layers
python algnet.py with baseline_configs/config_2x2.json n_hidden=5
```

### Adversarial attack simulation

This implementation also includes the option to simulate adversarial attacks against the auction learner.
There are two types of attack scenarios that can be simulated:
- Offline attack: The adversary chooses a bidding distribution prior to to taking part in the auction, and then samples their bids from that distribution, which may not represent their true valuation profile.
- Online attack: The adversary receives the outcome of every auction during training and can adapt their bidding strategy during every iteration of the auction.

#### Offline attack
To simulate an offline attack, set `attack_mode="offline"` as well as the `misreport_type` and the corresponding `misreport_params`.
Currently the offline attack only supports uniform and normal distributions.
```
# offline attack with uniformly distributed misreports
python algnet.py with baseline_configs/config_2x2.json attack_mode="offline" misreport_type="uniform" misreport_params="{'low': 0.0, 'high': 0.8}" num_steps=100
```
```
# offline attack with normally distributed misreports
python algnet.py with baseline_configs/config_2x2.json attack_mode="offline" misreport_type="normal" misreport_params="{'mean': 0.4, 'stddev': 0.2}" num_steps=100
```

#### Online attack
To simulate an online attack, set `attack_mode="online"`.
```
python algnet.py with baseline_configs/config_2x2.json attack_mode="online" num_steps=100
```

### Training with Differential Privacy

You can enable differentially private stochastic gradient descent (DPSGD) by setting `dp=True` during training.
```
python algnet.py with baseline_configs/config_2x2.json attack_mode="online" dp=True num_steps=1000
```
Initial tests suggest that this aids in achieving improved auction outcomes under attack conditions, but additional experiments are needed for confirmation.


## Logging and Artifacts

The project uses the [Sacred](https://github.com/IDSIA/sacred) framework for experiment tracking. 

- Logs and experiment metadata are saved to an SQLite database named `results.db`.
- The state parameters of the last trained model are saved to `tpal_state_params.pkl`.

For a quick overview over your past completed runs, you can use `db_inspect.py` to output summaries as a markdown table:
```
python db_inspect.py
```

## Implementation notes

This module is kept simple to make it suitable for use with computational experiment frameworks, or as a component in larger systems.
[Black](https://black.readthedocs.io/en/stable/) is used as a code formatter.

## Funding

This project is funded through the [NGI Assure Fund](https://nlnet.nl/assure), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) program. Learn more on the [NLnet project page](https://nlnet.nl/project/dist-mech-learn).
