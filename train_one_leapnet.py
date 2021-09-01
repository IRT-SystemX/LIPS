# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""
This script is used to train one augmented simulator (using a LeapNetAS) from command line.

It is meant to be used with "train_many.py" that uses the NeverGrad library to perform
automatic learning of some of these parameters (but requires training a lot of different models)

Usage:
> python3 train_one_leapnet.py --name 20210830_1504 --lr 0.0001551695351036826 --topo_vect_to_tau raw --size_main 92 --nb_layer_main 3 --size_enc 17 --nb_layer_enc 1 --size_dec 92 --nb_layer_dec 2 --nb_iter 500
"""

import json
import os
import argparse
from lips.neurips_benchmark import NeuripsBenchmark1
from tensorflow.keras.layers import Dense
from lips.augmented_simulators import LeapNetAS
import tensorflow as tf


def create_parser():
    this_parser = argparse.ArgumentParser()
    this_parser.add_argument("--nb_iter", type=int, default=200,
                             help="Number of training epoch")
    this_parser.add_argument("--name", type=str, default="test",
                             help="Name of the model")
    this_parser.add_argument("--lr", type=float, default=3e-4,
                             help="Learning rate")
    this_parser.add_argument("--layer_act", type=str, default="relu",
                             help="activation function [relu, leaky relu]")
    this_parser.add_argument("--loss", type=str, default="mse",
                             help="loss function used for training [mse]")
    this_parser.add_argument("--topo_vect_to_tau", type=str, default="raw",
                             help="loss function used for training [raw, online_list]")
    this_parser.add_argument("--batch_size", type=int, default=128,
                             help="Minibatch size")
    this_parser.add_argument("--mult_by_zero_lines_pred", type=bool, default=True,
                             help="Do you force the model to output 0 when powerlines are "
                                  "disconnected ? (default: True) -- not tested if this "
                                  "argument works. We do not recommend to change the default")

    # NN shape
    this_parser.add_argument("--size_main", type=int, default=300,
                             help="Number of units (per layer) for the 'main' layer")
    this_parser.add_argument("--nb_layer_main", type=int, default=3,
                             help="Number of layers for the 'main' layer")
    this_parser.add_argument("--size_enc", type=int, default=20,
                             help="Number of units (per layer) for the 'encoder' layer")
    this_parser.add_argument("--nb_layer_enc", type=int, default=3,
                             help="Number of layers for the 'encoder' layer")
    this_parser.add_argument("--size_dec", type=int, default=40,
                             help="Number of units (per layer) for the 'encoder' layer")
    this_parser.add_argument("--nb_layer_dec", type=int, default=2,
                             help="Number of layers for the 'encoder' layer")

    return this_parser


def main(name: str = "test",
         path_save=None,
         path_benchmark=None,
         sizes_layer=(300, 300, 300, 300),
         sizes_enc=(20, 20, 20),
         sizes_out=(100, 40),
         lr=3e-4,
         layer=Dense,
         layer_act="relu",
         loss="mse",
         batch_size: int = 128,
         nb_iter: int = 200,
         topo_vect_to_tau: str = "raw",
         kwargs_tau=None,
         mult_by_zero_lines_pred=True,
         command: str = None
         ):
    if path_benchmark is None:
        path_benchmark = os.path.join("reference_data")

    if path_save is None:
        path_save = os.path.join("trained_models")
        if not os.path.exists(path_save):
            os.mkdir(path_save)

    neurips_benchmark1 = NeuripsBenchmark1(path_benchmark=path_benchmark,
                                           load_data_set=True)

    # the three lines bellow might be familiar to the tensorflow users. They tell tensorflow to not take all
    # the GPU video RAM for the model.
    physical_devices = tf.config.list_physical_devices('GPU')
    for el in physical_devices:
        tf.config.experimental.set_memory_growth(el, True)

    my_simulator = LeapNetAS(name=name,
                             attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                             attr_tau=("topo_vect", "line_status"),
                             attr_y=("a_or", "a_ex"),
                             sizes_layer=sizes_layer,
                             sizes_enc=sizes_enc,
                             sizes_out=sizes_out,
                             lr=lr,
                             layer=layer,
                             layer_act=layer_act,
                             loss=loss,
                             batch_size=batch_size,
                             topo_vect_to_tau=topo_vect_to_tau,
                             kwargs_tau=kwargs_tau,
                             mult_by_zero_lines_pred=mult_by_zero_lines_pred)
    history_callback = my_simulator.train(nb_iter=nb_iter,
                                          train_dataset=neurips_benchmark1.train_dataset,
                                          val_dataset=neurips_benchmark1.val_dataset,
                                          verbose=0)
    # save the model
    my_simulator.save(path_save)
    my_simulator.save_metadata(path_save)
    # compute the evaluation on the validation set
    metrics_per_dataset = neurips_benchmark1.evaluate_augmented_simulator(my_simulator,
                                                                          dataset="val")
    # add
    metrics_per_dataset["neural net"] = {
        "sizes_layer": [int(el) for el in sizes_layer],
        "sizes_enc": [int(el) for el in sizes_enc],
        "sizes_out": [int(el) for el in sizes_out],
        "lr": float(lr),
        "layer_act": str(layer_act),
        "loss": str(loss),
        "batch_size": int(batch_size),
        "nb_iter": int(nb_iter),
        "topo_vect_to_tau": str(topo_vect_to_tau),
        "kwargs_tau": kwargs_tau,
        "command": command
    }
    metrics_per_dataset["training"] = history_callback.history
    save_json = os.path.join(path_save, name, "metrics.json")
    with open(save_json, encoding="utf-8", mode="w") as f:
        json.dump(fp=f, obj=metrics_per_dataset, indent=4, sort_keys=True)


if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    # command used to run the script
    command = ' '.join([sys.executable] + sys.argv)
    # define how the "topo_vect" will be converted into "tau"
    topo_vect_to_tau = str(args.topo_vect_to_tau)
    if topo_vect_to_tau == "raw":
        kwargs_tau = None
    elif topo_vect_to_tau == "online_list":
        kwargs_tau = 452  # TODO make that changeable (i put a "weird" number so that a ctrl+f
        # TODO finds it easily when a bug happens !)
    else:
        raise RuntimeError(f"Unrecognize topo_vect_to_tau argument. "
                           f"Found \"{topo_vect_to_tau}\" which is not supported."
                           "")

    # start the training
    main(nb_iter=int(args.nb_iter),
         name=str(args.name),
         lr=float(args.lr),
         layer_act=str(args.layer_act),
         loss=str(args.loss),
         batch_size=int(args.batch_size),
         sizes_layer=[int(args.size_main) for _ in range(args.nb_layer_main)],
         sizes_enc=[int(args.size_enc) for _ in range(args.nb_layer_enc)],
         sizes_out=[int(args.size_dec) for _ in range(args.nb_layer_dec)],
         topo_vect_to_tau=topo_vect_to_tau,
         kwargs_tau=kwargs_tau,
         command=command
         )
