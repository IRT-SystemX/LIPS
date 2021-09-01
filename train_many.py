# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""
This script is used to train (sequentially for now) multiple augmented simulator using the
"train_one_leapnet.py" script.

It uses nevergrad to perform an hyper parameter tuning.

Usage:
> python3 train_many.py --budget 200
"""

import datetime
import nevergrad as ng
import sys
import subprocess
import json
import os
import argparse

BUDGET = 200


def create_parser():
    this_parser = argparse.ArgumentParser()
    this_parser.add_argument("--budget", type=int, default=BUDGET,
                             help=f"Number of neural network that will be trained "
                                  f"default={BUDGET}")
    return this_parser


def train_model_seq(**kwargs):
    """train one model on one GPU"""
    name = f"{datetime.datetime.now():%Y%m%d_%H%M}"
    subprocess.run([sys.executable, "train_one_leapnet.py",
                    "--name", name,
                    "--lr", f'{kwargs["learning_rate"]}',
                    "--topo_vect_to_tau", kwargs["topo_vect_to_tau"],
                    "--size_main", f'{kwargs["size_main"]}',
                    "--nb_layer_main", f'{kwargs["nb_layer_main"]}',
                    "--size_enc", f'{kwargs["size_enc"]}',
                    "--nb_layer_enc", f'{kwargs["nb_layer_enc"]}',
                    "--size_dec", f'{kwargs["size_dec"]}',
                    "--nb_layer_dec", f'{kwargs["nb_layer_dec"]}',
                    "--nb_iter", "500",
                    ])
    with open(os.path.join("trained_models", name, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    val_loss = metrics["training"]["val_loss"][-1]
    return val_loss


def main(budget=BUDGET):
    # this code is inspired from the nevergrad documentation available at:
    # https://github.com/facebookresearch/nevergrad
    # https://facebookresearch.github.io/nevergrad/machinelearning.html#ask-and-tell-version

    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    parametrization = ng.p.Instrumentation(
        learning_rate=ng.p.Log(lower=1e-5, upper=1e-2),
        topo_vect_to_tau=ng.p.Choice(["raw", "online_list"]),
        size_main=ng.p.Scalar(lower=50, upper=300).set_integer_casting(),
        nb_layer_main=ng.p.Scalar(lower=1, upper=5).set_integer_casting(),
        size_enc=ng.p.Scalar(lower=5, upper=20).set_integer_casting(),
        nb_layer_enc=ng.p.Scalar(lower=0, upper=3).set_integer_casting(),
        size_dec=ng.p.Scalar(lower=20, upper=150).set_integer_casting(),
        nb_layer_dec=ng.p.Scalar(lower=0, upper=3).set_integer_casting(),
    )

    optim = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    for _ in range(budget):
        x1 = optim.ask()
        res = train_model_seq(**x1.kwargs)
        optim.tell(x1, res)
    recommendation = optim.recommend()
    train_model_seq(**recommendation.kwargs)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(budget=int(args.budget))
