
# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import copy
import pathlib
import time
import numpy as np
import tensorflow as tf

from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.augmented_simulators.tensorflow_models import LeapNet
from lips.dataset.scaler.powergrid_scaler import PowerGridScaler
from lips.config import ConfigManager


# indicate required paths
LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_case14_sandbox"
BENCH_CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
SIM_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "simulators"
BASELINES_PATH = LIPS_PATH / "trained_baselines" / "powergrid"
TRAINED_MODEL_PATH = LIPS_PATH / "trained_models" / "powergrid"
EVALUATION_PATH = LIPS_PATH / "evaluation_results" / "PowerGrid"
LOG_PATH = LIPS_PATH / "lips_logs.log"

benchmark1 = PowerGridBenchmark(benchmark_name="Benchmark1",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=BENCH_CONFIG_PATH
                               )

def test_fast_transform_tau():
    """
    Test the fast transform of tau vector
    """
    bench_config = ConfigManager(section_name="Benchmark1", path=BENCH_CONFIG_PATH)
    topo_actions = bench_config.get_option("dataset_create_params")["reference_args"]["topo_actions"]

    kwargs_tau = []
    for el in topo_actions:
         kwargs_tau.append(el["set_bus"]["substations_id"][0])

    leap_net1 = LeapNet(name="tf_leapnet",

                        bench_config_path=BENCH_CONFIG_PATH,
                        bench_config_name="Benchmark1",
                        sim_config_path=SIM_CONFIG_PATH / "tf_leapnet.ini",
                        sim_config_name="DEFAULT",
                        log_path=LOG_PATH,

                        loss={"name": "mse"},
                        lr=1e-4,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.01),

                        sizes_enc=(),
                        sizes_main=(150, 150),
                        sizes_out=(),
                        topo_vect_to_tau="given_list",
                        is_topo_vect_input=False,
                        kwargs_tau=kwargs_tau,
                        layer="resnet",
                        attr_tau=("line_status","topo_vect"),
                        scale_main_layer=150,
                        # scale_input_enc_layer = 40,
                        #scale_input_dec_layer=200,
                        # topo_vect_as_input = True,
                        mult_by_zero_lines_pred=False,
                        topo_vect_as_input=True,
                        scaler=PowerGridScaler,

                        )

    ## add topo_vect (temporary ) in attr_x in benchmark config file


    ## add topo_vect (temporary ) in attr_x in benchmark config file
    ##############
    nb_timesteps_to_test=1000
    indices=[i for i in range(nb_timesteps_to_test)]
    for key in benchmark1.train_dataset.data.keys():
        benchmark1.train_dataset.data[key]=benchmark1.train_dataset.data[key][indices]

    benchmark1.train_dataset.size=len(indices)

    leap_net1._leap_net_model.max_row_training_set=len(indices)
    dataset=benchmark1.train_dataset
    obss = leap_net1._make_fake_obs(dataset)
    leap_net1._leap_net_model.init(obss)


    (extract_x, extract_tau), extract_y = leap_net1.scaler.fit_transform(dataset)

    #####
    #Launch two different mathods for transformation and check that they match
    tau = copy.deepcopy(extract_tau)
    start = time.time()
    extract_tau_1 = leap_net1._transform_tau(dataset, extract_tau)
    end = time.time()
    print(end - start) #3.9s pour 10 000 / Pour 100 000 35.63s

    start = time.time()
    extract_tau_bis = leap_net1._transform_tau_given_list(tau)
    end = time.time()
    print(end - start) #0.026s pour 10 000 => 150 fois plus rapide! Pour 100 000, 1,7s
    #0.021s with int32, could we only work with boolean ? 0.32s for 100 000 (or 0.82s with numpy matmult, a bit faster with tensorflow then)

    assert np.all(extract_tau_1[1].astype((np.bool_))==extract_tau_bis[1])
