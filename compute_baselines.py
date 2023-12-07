import json
import subprocess
import itertools
import numpy as np

from scipy import stats

# Hyperparameter ranges
hyperparameter_ranges = {
    "net_depth": [3, 7],
    "net_width": [50, 100, 200],
    "learning_rate": [0.001, 0.0005],
    "num_steps": [160000, 240000],
    "misr_reinit_iv": [800, 1600],
    "misr_reinit_lim": [40000, 60000],
    "misr_updates": [100]
}

baseline_revenue_regret = {
    "1x2": {
        "revenue": (0.555, 0.0019),
        "regret": (0.00055, 0.00014)
    },
    "1x10": {
        "revenue": (3.487, 0.0135),
        "regret": (0.00165, 0.00057)
    },
    "2x2": {
        "revenue": (0.879, 0.0024),
        "regret": (0.00058, 0.00023)
    },
    "3x10": {
        "revenue": (5.562, 0.0308),
        "regret": (0.00193, 0.00033)
    },
    "5x10": {
        "revenue": (6.781, 0.0504),
        "regret": (0.00385, 0.00043)
    }
}

# some randomly chosen random seeds
rng_seeds = {
    "training" : [5100, 1529, 4889, 3234, 4375],
    "test":      [8298, 6266, 136, 2619, 4709]
}


def run_algnet_with_config(
        config,
        num_steps,
        misr_updates,
        misr_reinit_iv,
        misr_reinit_lim,
        net_depth,
        net_width,
        num_test_samples,
        learning_rate,
        seed_train,
        seed_test ):
    # Run algnet.py with the specified configuration file.
    cmd = f"python algnet.py with baseline_configs/config_{config}.json num_steps={num_steps} misr_updates={misr_updates} misr_reinit_iv={misr_reinit_iv} misr_reinit_lim={misr_reinit_lim} net_depth={net_depth} net_width={net_width} num_test_samples={num_test_samples} learning_rate={learning_rate} rng_seed_training={seed_train} rng_seed_test={seed_test}"
    print(f"starting training for config {config} with hyperparameters:")
    print(f"num_steps={num_steps}\nmisr_updates={misr_updates}\nmisr_reinit_iv={misr_reinit_iv}\nmisr_reinit_lim={misr_reinit_lim}\nnet_depth={net_depth}\nnet_width={net_width}\nnum_test_samples={num_test_samples}\nlearning_rate={learning_rate}\nrng_seed_training={seed_train}\nrng_seed_test={seed_test}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.stdout

def test_hyperparameters(config):
    # Iterating over hyperparameters in the order of assumed computational cost
    for num_steps in hyperparameter_ranges["num_steps"]:
        for net_depth in hyperparameter_ranges["net_depth"]:
            for net_width in hyperparameter_ranges["net_width"]:
                for lr in hyperparameter_ranges["learning_rate"]:
                    for misr_reinit_iv in hyperparameter_ranges["misr_reinit_iv"]:
                        for misr_reinit_lim in hyperparameter_ranges["misr_reinit_lim"]:
                            for i in range(5):
                                seed_train = rng_seeds["training"][i]
                                seed_test  = rng_seeds["test"][i]
                                run_algnet_with_config(
                                   config,
                                   num_steps,
                                   100, # misr_updates
                                   misr_reinit_iv,
                                   misr_reinit_lim,
                                   net_depth,
                                   net_width,
                                   10000, # num_test_samples
                                   lr,
                                   seed_train,
                                   seed_test
                               )
def test_chosen(config, choice_indices, training_seeds, test_seeds):
    n = len(training_seeds)
    assert(n==len(test_seeds))
    for i in range(n):
        num_steps       = hyperparameter_ranges["num_steps"      ][choice_indices[0]]
        net_depth       = hyperparameter_ranges["net_depth"      ][choice_indices[1]]
        net_width       = hyperparameter_ranges["net_width"      ][choice_indices[2]]
        lr              = hyperparameter_ranges["learning_rate"  ][choice_indices[3]]
        misr_reinit_iv  = hyperparameter_ranges["misr_reinit_iv" ][choice_indices[4]]
        misr_reinit_lim = hyperparameter_ranges["misr_reinit_lim"][choice_indices[5]]

        seed_train = training_seeds[i]
        seed_test  = test_seeds[i]

        run_algnet_with_config(
            config,
            num_steps,
            100, # misr_updates
            misr_reinit_iv,
            misr_reinit_lim,
            net_depth,
            net_width,
            10000, # num_test_samples
            lr,
            seed_train,
            seed_test
        )

test_chosen("1x2", [0,0,0,0,0,0,0], rng_seeds["training"], rng_seeds["test"])
