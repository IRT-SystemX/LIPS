# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import pathlib
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.augmented_simulators.tensorflow_models.powergrid.leap_net import LeapNet
from lips.dataset.scaler.powergrid_scaler import PowerGridScaler

# indicate required paths
LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_case14_sandbox"
BENCH_CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
SIM_CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "simulators"

benchmark1 = PowerGridBenchmark(benchmark_name="Benchmark1",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=None,
                                config_path=BENCH_CONFIG_PATH
                               )

benchmark2 = PowerGridBenchmark(benchmark_name="Benchmark2",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=None,
                                config_path=BENCH_CONFIG_PATH
                               )

def test_train_leapnet_raw():
    """Test if the training of the LeapNet can be executed as expected

    Test leap_net using
        ``topo_vect_to_tau="raw"``
    """

    leap_net = LeapNet(name="tf_leapnet",
                       bench_config_path=BENCH_CONFIG_PATH,
                       bench_config_name="Benchmark1",
                       sim_config_path=SIM_CONFIG_PATH / "tf_leapnet.ini",
                       sim_config_name="DEFAULT",
                       sizes_main=(150, 150),
                       sizes_enc=(20, 20, 20),
                       sizes_out=(100, 40),
                       scaler=PowerGridScaler,
                       topo_vect_to_tau="raw",
                       log_path=None)

    leap_net.train(train_dataset=benchmark1.train_dataset,
                   val_dataset=benchmark1.val_dataset,
                   epochs=2
                   )

    assert leap_net.trained is True

    leap_net = LeapNet(name="tf_leapnet",
                       bench_config_path=BENCH_CONFIG_PATH,
                       bench_config_name="Benchmark2",
                       sim_config_path=SIM_CONFIG_PATH / "tf_leapnet.ini",
                       sim_config_name="DEFAULT",
                       sizes_main=(150, 150),
                       sizes_enc=(20, 20, 20),
                       sizes_out=(100, 40),
                       scaler=PowerGridScaler,
                       topo_vect_to_tau="raw",
                       log_path=None)

    leap_net.train(train_dataset=benchmark2.train_dataset,
                   val_dataset=benchmark2.val_dataset,
                   epochs=2
                   )

    assert leap_net.trained is True
