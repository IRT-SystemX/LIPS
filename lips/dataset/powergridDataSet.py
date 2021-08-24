# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import warnings
import numpy as np
import shutil

import copy
from typing import Union

from grid2op.Environment import Environment
from grid2op.Agent import BaseAgent, DoNothingAgent
from lips.dataset.dataSet import DataSet


class PowerGridDataSet(DataSet):
    """
    This specific DataSet uses grid2op framework to simulate data coming from a powergrid.
    """

    def __init__(self,
                 experiment_name="Scenario",
                 # for compatibility with existing code this will be removed in future version
                 # (and serialize directly the output of the simulator)
                 attr_names=("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                             "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                             "v_or", "v_ex")

                 ):
        DataSet.__init__(self, experiment_name=experiment_name)
        self._nb_divergence = 0
        self._attr_names = copy.deepcopy(attr_names)

    def generate(self,
                 simulator: Environment,
                 actor: Union[None, BaseAgent],
                 path_out,
                 nb_samples,
                 simulator_seed: Union[None, int] =None,
                 actor_seed: Union[None, int] =None):
        """

        Parameters
        ----------
        simulator:
           In this case, this should be a grid2op environment

        actor:

        path_out:
            The path where the data will be saved

        nb_samples:
            Number of rows (examples) in the final dataset

        simulator_seed:
            Seed used to set the simulator for reproducible experiments

        actor_seed:
            Seed used to set the actor for reproducible experiments

        Returns
        -------

        """
        self._nb_divergence = 0
        if nb_samples <= 0:
            raise RuntimeError("Impossible to generate a negative number of data.")
        assert isinstance(simulator, Environment), "simulator should be a grid2op Environment"
        if actor is None:
            actor = DoNothingAgent(simulator.action_space)

        assert isinstance(actor, BaseAgent), "actor should be a grid2op Agent (type BaseAgent)"
        not_enough_data = True
        obs = simulator.reset()
        self.data = {}
        for attr_nm in self._attr_names:
            # this part is only temporary, until a viable way to store the complete resulting state is found
            array_ = getattr(obs, attr_nm)
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)

        from tqdm import tqdm  # TODO remove for final push
        for ds_size in tqdm(range(nb_samples)):
            done = True
            reward = simulator.reward_range[0]
            while done:
                # simulate data (resimulate in case of divergence of the simulator)
                act = actor.act(obs, reward, done)
                obs, reward, done, info = simulator.step(act)
                if info["is_illegal"]:
                    raise RuntimeError("Your `actor` should not take illegal action. Please modify the environment "
                                       "or your actor.")
                if done:
                    self._nb_divergence += 1
                    obs = simulator.reset()  # reset the simulator in case of "divergence"d
            self._store_obs(ds_size, obs)

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)

    def _store_obs(self, current_size, obs):
        """store an observation in self.data"""
        for attr_nm in self._attr_names:
            array_ = getattr(obs, attr_nm)
            self.data[attr_nm][current_size, :] = array_

    def _save_internal_data(self, path_out):
        """save the self.data in a proper format"""
        full_path_out = os.path.join(os.path.abspath(path_out), self.experiment_name)
        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))
            print(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")

        if os.path.exists(full_path_out):
            # deleting previous saved data
            print(f"Deleting previous run at {full_path_out}")
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)
        print(f"Creating the path {path_out} to store the dataset")

        for attr_nm in self._attr_names:
            np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data=self.data[attr_nm])

    def load(self, path):
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.experiment_name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            if not os.path.exists(path_this_array):
                raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                                   f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

        if self.data is not None:
            warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
        self.data = {}
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.data[attr_nm] = np.load(path_this_array)["data"]
