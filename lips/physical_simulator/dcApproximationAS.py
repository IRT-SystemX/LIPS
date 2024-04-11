# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union
from tqdm import tqdm
import time

import numpy as np


from grid2op.Backend import PandaPowerBackend
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action import CompleteAction
from lightsim2grid import LightSimBackend

from . import Grid2opSimulator
from . import PhysicsSolver
from ..dataset.powergridDataSet import PowerGridDataSet
from ..config import ConfigManager


class DCApproximationAS(PhysicsSolver):
    """
    This class presents a physics solver used for power system flow computation.

    It is based on a physical solver that linearized the powergrid equations and solve this linearization.

    Todo
    ----
    TODO : add logger and log the computation time

    # TODO : to remove this description
    The goal of this file is double. First it illustrates that there is no strict difference between an
    "AugmentedSimulator" and a "PhysicalSimulator" (one can be transformed into another). And secondly it serves
    as a example implementation for a specific `AugmentedSimulator`.

    """
    def __init__(self,
                 name: str = "dc_approximation",
                 benchmark_name: str = "Benchmark1",
                 config_path: Union[str, None] = None,
                 grid_path: Union[str, None] = None,
                 simulator: Union[Grid2opSimulator, None] = None,
                 backend: Union[PandaPowerBackend, LightSimBackend]=PandaPowerBackend,
                 is_dc: bool = True,
                 ignore_assert: bool = False
                 ):
        PhysicsSolver.__init__(self, name=name)
        self.config_manager = ConfigManager(path=config_path, section_name=benchmark_name)
        # input that will be given to the augmented simulator
        self._attr_x = ("prod_p", "prod_v", "load_p", "load_q", "topo_vect")
        # output that we want the proxy to predict
        #self._attr_y = ("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex")
        attr_y = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"}
        self._attr_y = tuple(set(self.config_manager.get_option("attr_y")).intersection(attr_y))
        # TODO : this attribute is not already used
        self._attr_fix_gen_p = "__prod_p_dc"

        if grid_path is not None:
            self._raw_grid_simulator = backend()
            self._raw_grid_simulator.load_grid(grid_path)
            if not(ignore_assert):
                self._raw_grid_simulator.assert_grid_correct()
        elif simulator is not None:
            assert isinstance(simulator, Grid2opSimulator), "To make the DC approximation augmented simulator, you " \
                                                            "should provide the reference grid as a Grid2opSimulator"

            self._raw_grid_simulator = simulator._simulator.backend.copy()
        else:
            raise RuntimeError("Impossible to initialize the DC approximation with a grid2op simulator or"
                               "a powergrid")

        self._bk_act_class = _BackendAction.init_grid(self._raw_grid_simulator)
        self._act_class = CompleteAction.init_grid(self._raw_grid_simulator)
        self.is_dc = is_dc
        self.comp_time = 0

    def compute(self, dataset: PowerGridDataSet):
        """
        evaluate the model on the whole dataset. It returns a dictionnary with the keys in self._attr_y
        """
        if not isinstance(dataset, PowerGridDataSet):
            raise RuntimeError("The DCApproximation can only be used with a PowerGridDataSet")

        nb_sample = len(dataset)
        res = {el: np.zeros((nb_sample, self._get_attr_size(el))) for el in self._attr_y}
        #res[self._attr_fix_gen_p] = np.zeros((nb_sample, self._get_attr_size("prod_p")))
        self.comp_time = 0
        for ind in tqdm(range(nb_sample), desc="evaluate dc"):
            # extract the current data
            data_this = dataset.get_data(np.array([ind], dtype=int))

            # process it
            self.process_dataset(data_this)

            # store the results
            tmp = {}
            tmp["p_or"], tmp["q_or"], tmp["v_or"], tmp["a_or"] = self._raw_grid_simulator.lines_or_info()
            tmp["p_ex"], tmp["q_ex"], tmp["v_ex"], tmp["a_ex"] = self._raw_grid_simulator.lines_ex_info()
            tmp1, tmp2, tmp["load_v"] = self._raw_grid_simulator.loads_info()
            tmp["prod_p"], tmp["prod_q"], tmp2 = self._raw_grid_simulator.generators_info()
            for attr_nm in self._attr_y:
                res[attr_nm][ind, :] = 1.0 * tmp[attr_nm]
            #res[self._attr_fix_gen_p][ind, :] = 1.0 * tmp["prod_p"]
        self._observations[dataset.name] = dataset.data
        self._flow[dataset.name] = res
        return res

    def _get_attr_size(self, attr_nm):
        if attr_nm == "a_or":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "a_ex":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "p_or":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "p_ex":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "q_or":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "q_ex":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "prod_q":
            return self._raw_grid_simulator.n_gen
        elif attr_nm == "load_v":
            return self._raw_grid_simulator.n_load
        elif attr_nm == "v_or":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "v_ex":
            return self._raw_grid_simulator.n_line
        elif attr_nm == "prod_p":
            return self._raw_grid_simulator.n_gen
        else:
            raise RuntimeError(f"Unknown attribute {attr_nm}")

    def init(self, **kwargs):
        """Initialization is done in the constructor for now"""
        pass

    def process_dataset(self, one_example):
        """
        this function process one data, by setting the solver of this class to the state observed
        in "one_example"
        """
        # modify the simulator
        modifer = self._bk_act_class()
        act = self._act_class()
        act.update({"set_bus": one_example["topo_vect"][0, :],
                    "injection": {
                        "prod_p": one_example["prod_p"][0, :],
                        "prod_v": one_example["prod_v"][0, :],
                        "load_p": one_example["load_p"][0, :],
                        "load_q": one_example["load_q"][0, :],
                    }
                    })
        modifer += act
        self._raw_grid_simulator.apply_action(modifer)

        # start the simulator
        if isinstance(self._raw_grid_simulator, PandaPowerBackend):
            _beg = time.time()
            self._raw_grid_simulator.runpf(is_dc=self.is_dc)
            self.comp_time += time.time() - _beg
        elif isinstance(self._raw_grid_simulator, LightSimBackend):
            beg_ = self._raw_grid_simulator._timer_solver
            self._raw_grid_simulator.runpf(is_dc=self.is_dc)
            self.comp_time += self._raw_grid_simulator._timer_solver - beg_

    def save(self, path_out):
        """
        this is not used for now.
        Ideally it should save the grid
        """
        pass

    def restore(self, path):
        """
        This is not used for now
        """
        pass
