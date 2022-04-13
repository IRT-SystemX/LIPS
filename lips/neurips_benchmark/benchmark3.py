# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import shutil
import numpy as np
import warnings
import copy
import logging

from lips.neurips_benchmark import NeuripsBenchmark1
from lips.benchmark.utils.scen1_utils import (get_kwargs_simulator_scenario,
                                                get_actor_training_scenario,
                                                get_actor_test_ood_topo_scenario,
                                                get_actor_test_scenario)

from lips.physical_simulator import Grid2opSimulator
from lips.dataset import PowerGridDataSet
from lips.evaluation import Evaluation

logging.basicConfig(filename="logs.log", level=logging.INFO,format="%(levelname)s:%(message)s")



class NeuripsBenchmark3(NeuripsBenchmark1):
    def __init__(self,
                 path_benchmark,
                 benchmark_name="NeuripsBenchmark3",
                 load_data_set=False,
                 train_env_seed: int = 1,
                 val_env_seed: int = 2,
                 test_env_seed: int = 3,
                 test_ood_topo_env_seed: int = 4,
                 initial_chronics_id: int = 0,
                 train_actor_seed: int = 5,
                 val_actor_seed: int = 6,
                 test_actor_seed: int = 7,
                 test_ood_topo_actor_seed: int = 8,
                 evaluation=None
                 ):
        NeuripsBenchmark1.__init__(self,
                                   path_benchmark,
                                   benchmark_name,
                                   load_data_set,
                                   train_env_seed,
                                   val_env_seed,
                                   test_env_seed,
                                   test_ood_topo_env_seed,
                                   initial_chronics_id,
                                   train_actor_seed,
                                   val_actor_seed,
                                   test_actor_seed,
                                   test_ood_topo_actor_seed,
                                   evaluation   
                                   )

        if evaluation is None:
            self.evaluation = Evaluation()
            # TODO fill that appropriately
            self.evaluation.active_dict["evaluate_ML"] = True
            self.evaluation.activate_physic_compliances()
            self.evaluation.active_dict["evaluate_adaptability"] = False
            self.evaluation.active_dict["evaluate_readiness"] = False
            self.evaluation.active_dict["evaluate_physic"]["verify_current_pos"] = True
            self.evaluation.active_dict["evaluate_physic"]["verify_voltage_pos"] = True
            self.evaluation.active_dict["evaluate_physic"]["verify_loss_pos"] = True
            self.evaluation.active_dict["evaluate_physic"]["verify_predict_disc"] = True
            self.evaluation.active_dict["evaluate_physic"]["verify_current_eq"] = True
            self.evaluation.active_dict["evaluate_physic"]["verify_EL"] = True
            self.evaluation.active_dict["evaluate_physic"]["verify_LCE"] = True
            self.evaluation.active_dict["evaluate_physic"]["verify_KCL"] = True
            #self.evaluation.active_dict["evaluate_physics"] = {}
        else:
            self.evaluate = evaluation

        self.train_dataset = PowerGridDataSet("train") # Integrate the theta variables in the analysis
        self.val_dataset = PowerGridDataSet("val")
        self._test_dataset = PowerGridDataSet("test")
        self._test_ood_topo_dataset = PowerGridDataSet("test_ood_topo")
        
        self.path_datasets = None
        if load_data_set:
            self.load()
        else:
            self.is_loaded = False