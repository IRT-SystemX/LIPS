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
from lips.neurips_benchmark.scen2_utils import (get_kwargs_simulator_scenario2,
                                                get_actor_training_scenario2,
                                                get_actor_test_ood_topo_scenario2,
                                                get_actor_test_scenario2)

from lips.physical_simulator import Grid2opSimulator
from lips.dataset import PowerGridDataSet
from lips.evaluation import Evaluation

logging.basicConfig(filename="logs.log", level=logging.INFO,format="%(levelname)s:%(message)s")



class NeuripsBenchmark2(NeuripsBenchmark1):
    def __init__(self,
                 path_benchmark,
                 benchmark_name="NeuripsBenchmark2",
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
            self.evaluation.active_dict["evaluate_physic"]["verify_KCL"] = False
            #self.evaluation.active_dict["evaluate_physics"] = {}

        else:
            self.evaluate = evaluation

    def _create_training_simulator(self):
        """"""
        if self.training_simulator is None:
            self.training_simulator = Grid2opSimulator(get_kwargs_simulator_scenario2(),
                                                       initial_chronics_id=self.initial_chronics_id,
                                                       # i use 994 chronics out of the 904 for training
                                                       chronics_selected_regex="^((?!(.*9[0-9][0-9].*)).)*$"
                                                       )

    def _fills_actor_simulator(self):
        """This function is only called when the data are simulated"""
        self._create_training_simulator()
        self.training_simulator.seed(self.train_env_seed)

        self.val_simulator = Grid2opSimulator(get_kwargs_simulator_scenario2(),
                                              initial_chronics_id=self.initial_chronics_id,
                                              # i use 50 full chronics for testing
                                              chronics_selected_regex=".*9[0-4][0-9].*")
        self.val_simulator.seed(self.val_env_seed)

        self.test_simulator = Grid2opSimulator(get_kwargs_simulator_scenario2(),
                                               initial_chronics_id=self.initial_chronics_id,
                                               # i use 25 full chronics for testing
                                               chronics_selected_regex=".*9[5-9][0-4].*")
        self.test_simulator.seed(self.test_env_seed)

        self.test_ood_topo_simulator = Grid2opSimulator(get_kwargs_simulator_scenario2(),
                                                        initial_chronics_id=self.initial_chronics_id,
                                                        # i use 25 full chronics for testing
                                                        chronics_selected_regex=".*9[5-9][5-9].*")
        self.test_ood_topo_simulator.seed(self.test_ood_topo_env_seed)

        self.training_actor = get_actor_training_scenario2(self.training_simulator)
        self.training_actor.seed(self.train_actor_seed)

        self.val_actor = get_actor_test_scenario2(self.val_simulator)
        self.val_actor.seed(self.val_actor_seed)

        self.test_actor = get_actor_test_scenario2(self.test_simulator)
        self.test_actor.seed(self.test_actor_seed)

        self.test_ood_topo_actor = get_actor_test_ood_topo_scenario2(self.test_ood_topo_simulator)
        self.test_ood_topo_actor.seed(self.test_ood_topo_actor_seed)