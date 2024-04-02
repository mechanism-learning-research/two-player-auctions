# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
# Copyright 2021 Daniel Reusche
# Copyright 2023 Tarek Sabet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on https://github.com/deepmind/dm-haiku/blob/4ae60fd4fd2da3b2f8f9ad3ec6dfd893745b483b/examples/mnist_gan.ipynb

import functools
import os
from datetime import datetime
from typing import Any, NamedTuple

import haiku as hk
import jax
import optax
import jax.numpy as jnp

from haiku.nets import MLP
import jax.nn as nn

from chex import assert_shape, assert_equal_shape

import joblib

from sacred import Experiment
from sacred.observers import SqlObserver

# Create a new experiment
ex = Experiment("auction_experiment")

# Attach an SQLite observer
ex.observers.append(SqlObserver("sqlite:///results.db"))


# Define configurations for the experiment
@ex.config
def cfg():
    num_steps = 1000  # Default value, can be overwritten when running the script
    num_test_samples = 10
    misr_updates = 50
    misr_reinit_iv = 500
    misr_reinit_lim = 1000
    batch_size = 100
    bidders = 2
    items = 2
    hidden_width = 50
    n_hidden = 2
    learning_rate = 0.001
    rng_seed_training = 1729
    rng_seed_test = 1337
    attack_mode = None  # Can be 'online' or 'offline' or None
    misreport_type = "uniform"  # Can be 'uniform' or 'normal'
    misreport_params = {}
    # val_dist = ...  # TODO: add when ready


# Uncomment to disable asserts
# chex.disable_asserts()


# Model
class BidSampler:
    def __init__(self, rng, bidders, items):
        self.bidders = bidders
        self.items = items

        self.key = rng

    def sample(self, num_samples):
        self.key, self.subkey = jax.random.split(self.key)

        sample = jnp.stack(
            [
                jax.random.uniform(self.subkey, (self.bidders, self.items))
                for _ in range(0, num_samples)
            ],
            axis=0,
        )
        return sample


class ValuationMisreporterOffline:
    def __init__(
        self, rng, bidders, items, misreport_type="uniform", misreport_params=None
    ):
        self.bidders = bidders
        self.items = items
        self.key = rng
        self.misreport_type = misreport_type
        self.misreport_params = misreport_params if misreport_params is not None else {}

    def misreport(self, val_samples):
        self.key, subkey = jax.random.split(self.key)
        modified_samples = []

        for val_sample in val_samples:
            # Sample for the first bidder (assumed misreporting)
            if self.misreport_type == "uniform":
                low = self.misreport_params.get("low", 0)
                high = self.misreport_params.get("high", 1)
                misreport_sample = jax.random.uniform(
                    subkey, (1, self.items), minval=low, maxval=high
                )
            elif self.misreport_type == "normal":
                mean = self.misreport_params.get("mean", 0)
                stddev = self.misreport_params.get("stddev", 1)
                misreport_sample = (
                    jax.random.normal(subkey, (1, self.items)) * stddev + mean
                )
            else:
                raise ValueError(f"Unsupported misreport type: {self.misreport_type}")

            # Replace the truthful sample for the first bidder with the misreported sample
            modified_sample = jnp.concatenate(
                [misreport_sample, val_sample[1:]], axis=0
            )
            modified_samples.append(modified_sample)

        return jnp.stack(modified_samples, axis=0)


class OnlineMisreporter(hk.Module):
    """Online Misreporter network using an MLP for generating misreports."""

    def __init__(self, bidders, items, hidden_width, n_hidden, name=None):
        super().__init__(name=name)
        self.bidders = bidders
        self.items = items
        self.hidden_width = hidden_width
        self.n_hidden = n_hidden

        input_width = self.bidders * self.items
        hidden_layers = [self.hidden_width] * self.n_hidden

        self.mlp = MLP([input_width, *hidden_layers, self.items], activation=jnp.tanh)

    def __call__(self, true_vals):
        misreports = self.mlp(jnp.ravel(true_vals))
        misreports = nn.sigmoid(misreports)  # Assuming valuations are in [0,1]
        return misreports


class ValuationMisreporterOnline:
    def __init__(self, rng, bidders, items, hidden_width, n_hidden, learning_rate):
        self.bidders = bidders
        self.items = items
        self.hidden_width = hidden_width
        self.n_hidden = n_hidden
        self.key = rng

        self.optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.999)

        # Define the Haiku network transform
        self.online_misreporter_transform = hk.without_apply_rng(
            hk.transform(
                lambda *args: OnlineMisreporter(
                    self.bidders, self.items, self.hidden_width, self.n_hidden
                )(*args)
            )
        )

        # Initialize the network and optimizer state
        self.params = self.online_misreporter_transform.init(
            self.key, jnp.zeros((self.bidders, self.items))
        )
        self.opt_state = self.optimizer.init(self.params)

    def misreport_single(self, val_sample):
        # Generate misreport for the first bidder using the MLP
        misreport = self.online_misreporter_transform.apply(self.params, val_sample)

        # Replace the truthful sample for the first bidder with the misreported sample
        modified_sample = val_sample.at[0].set(misreport)

        return modified_sample

    def misreport(self, val_batch):  # val_batch: (batch_size, bidders, items)
        v_misreport = jax.vmap(functools.partial(self.misreport_single))
        return v_misreport(val_batch)

    def utility_single(self, tpal, tpal_state, misreported_sample, val_sample):
        # Receive an auction from the current tpal_state using the misreported samples
        auct_params = tpal_state.params.auct
        alloc, pay = tpal.auct_transform.apply(auct_params, misreported_sample)

        # Calculate utility for the first bidder using their true valuations
        utility_first_bidder = alloc[0, :] @ val_sample[0, :] - pay[0]
        return utility_first_bidder

    def update(self, misreported_batch, val_batch, tpal, tpal_state):
        v_utility = jax.vmap(functools.partial(self.utility_single, tpal, tpal_state))
        utility_first_bidder_batch = v_utility(misreported_batch, val_batch)

        # Define the loss function
        def loss_fn(params):
            utility = jnp.mean(utility_first_bidder_batch)
            return -utility

        # Update the misreporter model using Optax
        grads = jax.grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.params
        )
        self.params = optax.apply_updates(self.params, updates)


# move b_i to the front of B
# B = [b_i, b_0, ..., b_i-1, b_i+1, ..., b_n]
def permute_along_bidders(B, i):
    if B.ndim == 1:
        return B

    head = B[:, 0:i]  # all bid profiles up to b_i
    tail = B[:, i + 1 :]  # all bid profiles after b_i
    b_i = B[:, i : i + 1]  # b_i, slice this way to preserve shape
    permuted = jnp.concatenate([b_i, head, tail], axis=1)

    assert_equal_shape([B, permuted])
    return permuted


class Auctioneer(hk.Module):
    """Auctioneer network."""

    def __init__(self, bidders, items, hidden_width, n_hidden, name=None):
        super().__init__(name=name)
        self.bidders = bidders
        self.items = items
        self.hidden_width = hidden_width
        self.n_hidden = n_hidden

        input_width = self.bidders * self.items
        hidden_layers = [self.hidden_width] * self.n_hidden

        # Layers for allocation MLPs
        alloc_layers = [input_width, *hidden_layers, self.items]
        # Layers for payment MLP
        pay_layers = [input_width, *hidden_layers, 1]

        # Initialize MLPs
        self.alloc_prob = MLP(alloc_layers, activation=jnp.tanh)
        self.alloc_which = MLP(alloc_layers, activation=jnp.tanh)
        self.pay_mlp = MLP(pay_layers, activation=jnp.tanh)

    def __call__(self, vals):
        """Computes auctions, consisting of an allocation and a payment matrix."""

        # rows are bidders
        # columns are items

        # probability to allocate an item
        alloc = self.alloc_prob(jnp.ravel(vals))
        alloc = nn.sigmoid(alloc)
        assert_shape(alloc, (self.items,))

        # probability to allocate item j to bidder i
        L = jnp.stack(  # stack bidder vectors to get matrix
            # compute bidder vectors
            [
                self.alloc_which(jnp.ravel(permute_along_bidders(vals, i)))
                for i in range(self.bidders)
            ],
            axis=0,
        )

        # softmax to ensure feasibility (allocate every item at most once).
        L = nn.softmax(L, axis=0)
        assert_shape(L, (self.bidders, self.items))

        alloc = alloc * L
        assert_shape(alloc, (self.bidders, self.items))

        # fraction of utility each bidder pays to the mechanism
        pay = jnp.squeeze(
            jnp.stack(
                [
                    nn.sigmoid(self.pay_mlp(jnp.ravel(permute_along_bidders(vals, i))))
                    for i in range(self.bidders)
                ],
                axis=1,
            )
        )

        # Fix shape for single bidder case.
        if self.bidders == 1:
            pay = jnp.stack([pay])

        assert_shape(pay, (self.bidders,))

        # fractions of utilities * sum of allocations of all items
        # per bidder for a given bid profile
        pay = pay * jnp.sum(jnp.squeeze(vals) * alloc, axis=1)
        # NOTE: squeeze vals, since they are in a batch of size one

        assert_shape(pay, (self.bidders,))
        return alloc, pay


class Misreporter(hk.Module):
    """Misreporter network."""

    def __init__(self, bidders, items, hidden_width, n_hidden, name=None):
        super().__init__(name=name)
        self.bidders = bidders
        self.items = items
        self.hidden_width = hidden_width
        self.n_hidden = n_hidden

        # Layers for misreporter MLP
        input_width = self.bidders * self.items
        hidden_layers = [self.hidden_width] * self.n_hidden
        misr_layers = [input_width, *hidden_layers, self.items]

        # Initialize MLP
        self.misr_mlp = MLP(misr_layers, activation=jnp.tanh)

    def __call__(self, vals):
        """Computes (approximately) optimal misreports for a given auction."""

        # TODO: JAXize more?
        m_ = []
        for i in range(self.bidders):
            misr = self.misr_mlp(jnp.ravel(permute_along_bidders(vals, i)))
            m_.append(misr)

        misreports = jnp.stack(m_)
        assert_shape(misreports, (self.bidders, self.items))

        # NOTE: sigmoid for [0,1] valuations, should be e.g. softplus for positive valuations
        misreports = nn.sigmoid(misreports)
        return misreports


def tree_shape(xs):
    return jax.tree_map(lambda x: x.shape, xs)


class TPALTuple(NamedTuple):
    auct: Any
    misr: Any


class TPALState(NamedTuple):
    params: TPALTuple
    opt_state: TPALTuple


class TPAL:
    """Two Player Auction Learner."""

    def __init__(self, bidders, items, hidden_width, n_hidden, learning_rate):
        self.bidders = bidders
        self.items = items

        self.hidden_width = hidden_width
        self.n_hidden = n_hidden

        # Define the Haiku network transforms.
        # We don't use BatchNorm so we don't use `with_state`.
        self.auct_transform = hk.without_apply_rng(
            hk.transform(
                lambda *args: Auctioneer(
                    self.bidders, self.items, self.hidden_width, self.n_hidden
                )(*args)
            )
        )

        self.misr_transform = hk.without_apply_rng(
            hk.transform(
                lambda *args: Misreporter(
                    self.bidders, self.items, self.hidden_width, self.n_hidden
                )(*args)
            )
        )

        # Build the optimizers. We use differentially private SGD.
        self.optimizers = TPALTuple(
            auct=optax.contrib.dpsgd(learning_rate, 10.07, 0.9, 1337, 0.9, True),
            misr=optax.contrib.dpsgd(learning_rate, 1.32,  0.9, 2342, 0.9, True),
        )


    @functools.partial(jax.jit, static_argnums=0)
    def initial_state(self, rng, vals):
        """Returns the initial parameters and optimize states."""
        # Get initial network parameters.
        rng, rng_auct, rng_misr = jax.random.split(rng, 3)

        params = TPALTuple(
            auct=self.auct_transform.init(rng_auct, vals),
            misr=self.misr_transform.init(rng_misr, vals),
        )

        def print_layers(params):
            for key, value in params.items():
                print(f"{key}:\tb = {value['b']}\tw = {value['w']}")

        print("Auctioneer:")
        print_layers(tree_shape(params.auct))

        print("\nMisreporter:")
        print_layers(tree_shape(params.misr))

        # Initialize the optimizers.
        opt_state = TPALTuple(
            auct=self.optimizers.auct.init(params.auct),
            misr=self.optimizers.misr.init(params.misr),
        )
        return TPALState(params=params, opt_state=opt_state)

    @functools.partial(jax.jit, static_argnums=0)
    def reinit_misr(self, rng, tpal_state, vals):
        """Reinitializes the misreporter."""

        # Get initial network parameters.
        rng, rng_misr = jax.random.split(rng)

        params = TPALTuple(
            auct=tpal_state.params.auct,
            misr=self.misr_transform.init(rng_misr, vals),
        )

        # Initialize the optimizers.
        opt_state = TPALTuple(
            auct=tpal_state.opt_state.auct,
            misr=self.optimizers.misr.init(params.misr),
        )
        return TPALState(params=params, opt_state=opt_state)

    # Calculate utilities for all players
    def utility(self, vals, alloc, pay):
        utilities = jnp.sum(jnp.squeeze(vals) * alloc, axis=1) - pay

        assert_equal_shape([utilities, pay])
        return utilities

    # check of utility[i] == utility_i
    def utility_i(self, vals, i, alloc, pay):
        return jnp.sum(alloc[i] * vals[i]) - pay[i]

    # Take misreports of bidder i while keeping the rest fixed
    def misr_bidder_i(self, vals, misrs, i):
        # In case of a single bidder, return the misreports directly
        if self.bidders == 1:
            return misrs

        # Create a boolean mask for the ith column
        mask = jnp.array([index == i for index in range(vals.shape[1])])

        # Select misreports for the ith bidder and original values for others
        V_minus_i = jnp.where(mask, misrs[:, i : i + 1], vals)

        return V_minus_i

    def misr_utility(self, misreports, val_sample, auct_params):
        # TODO: JAXize more?
        misr_utils = []
        for i in range(0, self.bidders):
            misr_i = self.misr_bidder_i(val_sample, misreports, i)
            assert_shape(misr_i, (self.bidders, self.items))

            # Receive an auction for misr_i
            alloc_m, pay_m = self.auct_transform.apply(auct_params, misr_i)

            u_i = self.utility(val_sample, alloc_m, pay_m)
            u_i = u_i[i]
            assert_shape(u_i, ())

            misr_utils.append(u_i)

        u_misr = jnp.stack(misr_utils)
        return u_misr

    def auct_loss(self, auct_params, misr_params, val_sample):
        """Auctioneer loss."""

        # Receive an auction
        alloc, pay = self.auct_transform.apply(auct_params, val_sample)
        # Receive misreports
        misreports = self.misr_transform.apply(misr_params, val_sample)

        regret = nn.relu(
            self.misr_utility(misreports, val_sample, auct_params)
            - self.utility(val_sample, alloc, pay)
        )

        loss = -(jnp.sqrt(jnp.sum(pay)) - jnp.sqrt(jnp.sum(regret))) + jnp.sum(regret)

        return loss

    def misr_loss(self, misr_params, auct_params, val_sample):
        """Misreporter loss."""

        # Receive misreports
        misreports = self.misr_transform.apply(misr_params, val_sample)

        # Calculate utility for misreports
        u_misr = self.misr_utility(misreports, val_sample, auct_params)

        return -jnp.sum(u_misr)

    # Vectorize losses to use on batches
    def v_auct_loss(self, auct_params, misr_params, val_batch):
        v_al = jax.vmap(functools.partial(self.auct_loss, auct_params, misr_params))
        return jnp.mean(v_al(val_batch))

    def v_misr_loss(self, misr_params, auct_params, val_batch):
        v_ml = jax.vmap(functools.partial(self.misr_loss, misr_params, auct_params))
        return jnp.mean(v_ml(val_batch))

    @functools.partial(jax.jit, static_argnums=0)
    def update_auct(self, tpal_state, batch):
        """Performs a parameter update."""
        # Update the generator.
        auct_loss = self.v_auct_loss(
            tpal_state.params.auct, tpal_state.params.misr, batch
        )

        # Uses jax.vmap across the batch to extract per-example gradients.
        grad_fn = jax.vmap(jax.grad(self.auct_loss), in_axes=(None, None, 0))
        auct_grads = grad_fn(tpal_state.params.auct, tpal_state.params.misr, batch)

        # Concatenate and flatten all bias and weight gradients
        all_gradients = jnp.concatenate([jnp.ravel(gradients['b']) for gradients in auct_grads.values()] +
                                [jnp.ravel(gradients['w']) for gradients in auct_grads.values()])

        # Compute the total gradient norm
        total_gradient_norm = jnp.linalg.norm(all_gradients)

        auct_update, auct_opt_state = self.optimizers.auct.update(
            auct_grads, tpal_state.opt_state.auct, tpal_state.params.auct
        )
        auct_params = optax.apply_updates(tpal_state.params.auct, auct_update)

        params = TPALTuple(auct=auct_params, misr=tpal_state.params.misr)
        opt_state = TPALTuple(auct=auct_opt_state, misr=tpal_state.opt_state.misr)
        tpal_state = TPALState(params=params, opt_state=opt_state)
        log = {
            "auct_loss": auct_loss,
            "auct_grad_norm": total_gradient_norm
        }
        return tpal_state, log

    @functools.partial(jax.jit, static_argnums=0)
    def update_misr(self, tpal_state, batch):
        """Performs a parameter update."""
        # Update the misreporter.
        misr_loss = self.v_misr_loss(
            tpal_state.params.misr, tpal_state.params.auct, batch
        )  # NOTE: Params of the network to be updated need to be the first arg.

        # Uses jax.vmap across the batch to extract per-example gradients.
        grad_fn = jax.vmap(jax.grad(self.misr_loss), in_axes=(None, None, 0))
        misr_grads = grad_fn(tpal_state.params.misr, tpal_state.params.auct, batch)

        # Concatenate and flatten all bias and weight gradients
        all_gradients = jnp.concatenate([jnp.ravel(gradients['b']) for gradients in misr_grads.values()] +
                                [jnp.ravel(gradients['w']) for gradients in misr_grads.values()])

        # Compute the total gradient norm
        total_gradient_norm = jnp.linalg.norm(all_gradients)

        misr_update, misr_opt_state = self.optimizers.misr.update(
            misr_grads, tpal_state.opt_state.misr, tpal_state.params.misr
        )
        misr_params = optax.apply_updates(tpal_state.params.misr, misr_update)

        params = TPALTuple(misr=misr_params, auct=tpal_state.params.auct)
        opt_state = TPALTuple(misr=misr_opt_state, auct=tpal_state.opt_state.auct)
        tpal_state = TPALState(params=params, opt_state=opt_state)
        log = {
            "misr_loss": misr_loss,
            "misr_grad_norm": total_gradient_norm
        }
        return tpal_state, log


# Train a two player auction learner and return it with state.
@ex.capture  # sacred experiment tracking decoration
def training(
    _run,  # for sacred logging
    num_steps,
    misr_updates,
    misr_reinit_iv,
    misr_reinit_lim,
    batch_size,
    bidders,
    items,
    hidden_width,
    n_hidden,
    learning_rate,
    rng_seed_training,
    attack_mode,
    misreport_type,
    misreport_params
    # val_dist, TODO: add option to use different distributions
):
    # @title {vertical-output: true}

    log_every = num_steps // 100

    # The model.
    tpal = TPAL(bidders, items, hidden_width, n_hidden, learning_rate)

    # Top-level RNG.
    rng = jax.random.PRNGKey(rng_seed_training)

    # Initialize the network and optimizer.
    rng, rng_sampler, rng_state_init, rng_misr_reinit, rng_val_misr = jax.random.split(
        rng, 5
    )

    bid_sampler = BidSampler(rng_sampler, bidders, items)

    tpal_state = tpal.initial_state(rng_state_init, bid_sampler.sample(1)[0])

    auct_grad_norms, misr_grad_norms = [], []

    valuation_misreporter = None
    match attack_mode:
        case "offline":
            valuation_misreporter = ValuationMisreporterOffline(
                rng_val_misr,
                bidders,
                items,
                misreport_type=misreport_type,
                misreport_params=misreport_params,
            )
        case "online":
            valuation_misreporter = ValuationMisreporterOnline(
                rng_val_misr, bidders, items, hidden_width, n_hidden, learning_rate
            )

    for step in range(num_steps):
        # Sample valuations using bid_sampler
        val_sample = bid_sampler.sample(batch_size)

        received_sample = (
            valuation_misreporter.misreport(val_sample)
            if valuation_misreporter
            else val_sample
        )

        if ((step % misr_reinit_iv) == 0) and (step <= misr_reinit_lim):
            tpal_state = tpal.reinit_misr(
                rng_misr_reinit, tpal_state, bid_sampler.sample(1)[0]
            )

        for _ in range(0, misr_updates):
            tpal_state, misr_log = tpal.update_misr(tpal_state, received_sample)

        tpal_state, auct_log = tpal.update_auct(tpal_state, received_sample)

        if attack_mode == "online":
            valuation_misreporter.update(received_sample, val_sample, tpal, tpal_state)

        auct_grad_norms.append(auct_log["auct_grad_norm"])
        misr_grad_norms.append(misr_log["misr_grad_norm"])

        # Log the losses.
        if step % log_every == 0:
            # It's important to call `device_get` here so we don't take up device
            # memory by saving the losses.
            misr_log = jax.device_get(misr_log)
            auct_log = jax.device_get(auct_log)

            auct_loss = auct_log["auct_loss"]
            misr_loss = misr_log["misr_loss"]
            print(
                f"Step {step}: "
                f"auct_loss = {auct_loss:.3f}, misr_loss = {misr_loss:.3f}, auct_grad_norm = {auct_log['auct_grad_norm']:.3f}, misr_grad_norm = {misr_log['misr_grad_norm']:.3f}"
            )

            # Logging Losses
            _run.log_scalar("losses.auct_loss", auct_loss, step)
            _run.log_scalar("losses.misr_loss", misr_loss, step)

    
    print("Median Auct Grad Norm:", jnp.median(jnp.array(auct_grad_norms)))
    print("Median Misr Grad Norm:", jnp.median(jnp.array(misr_grad_norms)))

    return tpal, tpal_state


# TODO vectorize and process all samples in parallel
def test(tpal, tpal_state, num_samples, rng_seed_test):
    rng = jax.random.PRNGKey(rng_seed_test)
    sampler = BidSampler(rng, tpal.bidders, tpal.items)

    truth_utils = []
    misr_utils = []
    regrets = []
    pays = []

    for _ in range(num_samples):
        val_sample = sampler.sample(1)

        # Receive an auction
        alloc, pay = tpal.auct_transform.apply(tpal_state.params.auct, val_sample)

        # Receive misreports
        misreports = tpal.misr_transform.apply(tpal_state.params.misr, val_sample)

        misr_util = tpal.misr_utility(
            misreports, jnp.squeeze(val_sample), tpal_state.params.auct
        )
        truth_util = tpal.utility(val_sample, alloc, pay)

        # check if this value is too negative, to see whether misreporter didn't converge
        # raw_regret = misr_util - truth_util
        regret = nn.relu(misr_util - truth_util)

        # Store results
        truth_utils.append(truth_util)
        misr_utils.append(misr_util)
        regrets.append(regret)
        pays.append(pay)

    return {
        "truth_util": jnp.stack(truth_utils),
        "misr_util": jnp.stack(misr_utils),
        "regret": jnp.stack(regrets),
        "pay": jnp.stack(pays),
    }


@ex.automain
def run(_run, _config):
    # Let's see what hardware we're working with. The training takes a few
    # minutes on a GPU, a bit longer on CPU.
    print("### Device information")
    print(f"Number of devices: {jax.device_count()}")
    print("Device:", jax.devices()[0].device_kind)
    print("")

    # Logging Device Information
    _run.log_scalar("devices.count", jax.device_count())
    _run.log_scalar("device.kind", str(jax.devices()[0].device_kind))
    _run.log_scalar("rng.seed.training", _config["rng_seed_training"])
    _run.log_scalar("rng.seed.test", _config["rng_seed_test"])

    # Logging Misreport Settings
    print("### Misreport Settings")
    print(f"Attack Mode: {_config['attack_mode']}")
    match _config["attack_mode"]:
        case "offline":
            print(f"Misreport Type: {_config['misreport_type']}")
            print(f"Misreport Parameters: {_config['misreport_params']}")
        case "online":
            pass
        case None:
            pass

    # Logging Misreport Settings
    _run.log_scalar("misreport.type", _config["misreport_type"])
    for param, value in _config["misreport_params"].items():
        _run.log_scalar(f"misreport.params.{param}", value)
    _run.log_scalar("attackmode", _config["attack_mode"])

    # Training the auctioneer
    print("### Starting training")
    bid_sampler = (
        ValuationMisreporterOffline
        if _config["attack_mode"] == "offline"
        else ValuationMisreporterOnline
    )
    tpal, tpal_state = training()  # no need to pass parameters explicitly

    # Serialize and save the TPAL model state
    if not os.path.exists("tpal_state_params"):
        os.makedirs("tpal_state_params")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_params_filename = f"tpal_state_params/{timestamp}.pkl"
    joblib.dump(tpal_state.params, state_params_filename)

    # Add the model and its state as artifacts to the Sacred run
    ex.add_artifact(state_params_filename)

    # Testing the auctioneer
    print("### Starting test")
    num_samples = _config["num_test_samples"]
    results = test(tpal, tpal_state, num_samples, _config["rng_seed_test"])

    print(f"### Average test results ({num_samples} samples)")
    averages = {}
    for key, matrix in results.items():
        # Save the matrix to a temporary file
        temp_filename = f"temp_{key}.pkl"
        joblib.dump(matrix, temp_filename)

        # Add the saved file as an artifact to the run
        _run.add_artifact(temp_filename, name=key)

        # Delete the temporary file after adding it to avoid clutter
        os.remove(temp_filename)

        # Log the averages
        total_values = jnp.sum(matrix, axis=1)
        average_total_value = jnp.mean(total_values)
        _run.log_scalar(f"avg_{key}", average_total_value)
        print(f"{key}: {average_total_value}")
        averages[key] = average_total_value

    avg_score = jnp.sqrt(averages["pay"]) - jnp.sqrt(averages["regret"])
    print(f"score: {avg_score}")
