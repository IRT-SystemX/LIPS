""" This module is used to generate powergrid datasets

Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

import os
import warnings
import shutil
import copy
from typing import Union, Callable
import numpy as np
from tqdm import tqdm  # TODO remove for final push

from grid2op.Agent import BaseAgent

from . import DataSet
from ..logger import CustomLogger
from ..physical_simulator import Grid2opSimulator

class PowerGridDataSet(DataSet):
    """Class to manage powergrid data

    This specific DataSet uses grid2op framework to simulate data coming from a powergrid.

    Attributes
    ----------
    name : ``str``, optional
        the dataset name, by default "train"
    attr_names : Union[``tuple``, ``None``], optional
        the attributes list for which data should be generated, by default None
    log_path : Union[``str``, ``None``], optional
        the path where the logs should be stored, by default None

    Todo
    -------
    TODO: to remove all the comments lines corresponding to theta attributes

    Examples
    --------

    .. code-block:: python

        from lips.dataset import PowerGridDataSet

        # create a dataset
        dataset = PowerGridDataSet()
        dataset.generate_data()

    """
    ALL_VARIABLES = ("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                     "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                     "v_or", "v_ex", "theta_or", "theta_ex")

    def __init__(self,
                 name: str="train",
                 # for compatibility with existing code this will be removed in future version
                 # (and serialize directly the output of the simulator)
                 attr_names: Union[tuple, None]=None,
                 log_path: Union[str, None]=None
                 ):
        DataSet.__init__(self, name=name)
        self._nb_divergence = 0
        if attr_names is not None:
            self._attr_names = copy.deepcopy(attr_names)
        else:
            self._attr_names = self.ALL_VARIABLES
        self.size = 0
        #self._theta_attr_names = copy.deepcopy(theta_attr_names)

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger

        # for the sampling
        self._previous = None
        self._order = None

        #TODO add a seed for reproducible experiment !

    def generate(self,
                 simulator: Grid2opSimulator,
                 actor: Union[BaseAgent, None],
                 path_out: Union[str, None],
                 nb_samples: int,
                 simulator_seed: Union[None, int] = None,
                 actor_seed: Union[None, int] = None):
        """Generate a powergrid dataset

        For this dataset, we use a Grid2opSimulator and a  grid2op Agent to generate data from a powergrid.
        This implementation can also serve as a reference for other implementation of the `generate` function.

        Parameters
        ----------
        simulator : Grid2opSimulator
            In this case, this should be a grid2op environment
        actor : Union[``BaseAgent``, ``None``]
            the actor used to generate the data
        path_out : Union[``str``, ``None``]
            The path where the data will be saved
        nb_samples : ``int``
            Number of rows (examples) in the final dataset
        simulator_seed : Union[``None``, ``int``], optional
            Seed used to set the simulator for reproducible experiments, by default None
        actor_seed : Union[``None``, ``int``], optional
            Seed used to set the actor for reproducible experiments, by default None

        Raises
        ------
        RuntimeError
            Impossible to generate powergird data, Grid2Op is not installed
        RuntimeError
            Impossible to generate negative number of data

        """
        try:
            from grid2op.Agent import DoNothingAgent
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` powergrid data if you don't have "
                               "the grid2Op package installed") from exc_
        self._nb_divergence = 0
        if nb_samples <= 0:
            raise RuntimeError("Impossible to generate a negative number of data.")

        # check that the proper data types are received
        super().generate(simulator, actor, path_out, nb_samples, simulator_seed, actor_seed)

        if actor is None:
            # TODO refactoring this, this is weird here
            actor = DoNothingAgent(simulator._simulator.action_space)

        init_state, init_info = simulator.get_state()
        self.data = {}
        for attr_nm in self._attr_names:
            # this part is only temporary, until a viable way to store the complete resulting state is found
            array_ = getattr(init_state, attr_nm)
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)

        # TODO : to be removed after verification
        """
        for idx_, attr_nm in enumerate(self._theta_attr_names):
            # this part is added to report the theta values
            array_ = init_theta[idx_]
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)
        """
        for ds_size in tqdm(range(nb_samples), desc=self.name):
            simulator.modify_state(actor)
            current_state, extra_info = simulator.get_state()
            self._store_obs(ds_size, current_state)
            #self._store_theta(ds_size, current_theta)

        self.size = nb_samples
        self._init_sample()
        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)

    def _store_obs(self, current_size, obs):
        """store an observation in self.data"""
        for attr_nm in self._attr_names:
            array_ = getattr(obs, attr_nm)
            self.data[attr_nm][current_size, :] = array_
    # TODO to be remove after verification
    """
    def _store_theta(self, current_size, theta):
        # store the theta (angles) in self.data
        for idx_, attr_nm in enumerate(self._theta_attr_names):
            self.data[attr_nm][current_size, :] = theta[idx_]
    """
    def _save_internal_data(self, path_out:str):
        """save the self.data in a proper format

        Parameters
        ----------
        path_out : ``str``
            path to save the data

        """
        full_path_out = os.path.join(os.path.abspath(path_out), self.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))
            # TODO logger
            #print(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")
            self.logger.info(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")

        if os.path.exists(full_path_out):
            # deleting previous saved data
            # TODO logger
            #print(f"Deleting previous run at {full_path_out}")
            self.logger.warning(f"Deleting previous run at {full_path_out}")
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)
        # TODO logger
        #print(f"Creating the path {full_path_out} to store the dataset name {self.name}")
        self.logger.info(f"Creating the path {full_path_out} to store the dataset name {self.name}")

        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        for attr_nm in self._attr_names:
            np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data=self.data[attr_nm])

    def load(self, path:str):
        """load the dataset from a path

        Parameters
        ----------
        path : ``str``
            path from which the data should be loaded

        Raises
        ------
        RuntimeError
            Path cannot be found
        RuntimeError
            Not a valid directory
        RuntimeError
            There is no data to load
        RuntimeError
            Impossible to load the data

        """
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")
        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            if not os.path.exists(path_this_array):
                raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                                   f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

        if self.data is not None:
            warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
        self.data = {}
        self.size = None
        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.data[attr_nm] = np.load(path_this_array)["data"]
            self.size = self.data[attr_nm].shape[0]

        self._init_sample()

    def _init_sample(self):
        """initialize the sample

        Init the sample
        """
        self._previous = 0
        self._order = np.arange(self.size)
        np.random.shuffle(self._order)

    def sample(self, nb_sample: int, sampler: Callable=None):
        """Sampling from the dataset

        For now, this sampling method will sample uniformly at random from the dataset.

        There is a guarantee: if you generate `sef.size` consecutive data with this method
        for example `batch1 = dataset.sample(data.size / 2)` then `batch2 = dataset.sample(data.size / 2)`
        in this case batch1 and batch2 will count different example of the dataset and the union batch1 and batch2
        will make the entire dataset [NB the above is true if the dataset has just been created, or if batch1
        comes from the first batch issued from this dataset.]

        Parameters
        ----------
        nb_sample : ``int``
            Number of sample to retrieve from the dataset.

        sampler : Callable
            currently unused

        Returns
        -------
        dict
            A batch of data, either for training or for testing.

        Raises
        ------
        RuntimeError
            Impossible to require a negative number of data
        RuntimeError
            Impossible to require more than the size of the dataset

        """
        if nb_sample < 0:
            raise RuntimeError("Impossible to require a negative number of data.")
        if nb_sample > self.size:
            raise RuntimeError("Impossible to require more than the size of the dataset")

        res = {}
        if nb_sample + self._previous < self.size:
            # i just sample the next batch of data
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el] = self.data[el][self._order[self._previous:(self._previous+nb_sample)], :]
            self._previous += nb_sample
        else:
            this_sz = self.size - self._previous
            # init the results
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)
            # fill with the remaining of the data
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el][:this_sz] = self.data[el][self._order[self._previous:], :]

            # sample another order to see the data
            self._init_sample()
            # fill with the remaining of the data
            self._previous = nb_sample - this_sz
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el][this_sz:] = self.data[el][self._order[:self._previous], :]
        return res

    def get_data(self, index: tuple) -> dict:
        """This function returns the data in the data that match the index `index`

        Parameters
        ----------
        index : ``tuple``
            A list of integer

        Returns
        -------
        dict
            a dictionary of key (variable name) value (variable value)

        """
        super().get_data(index)  # check that everything is legit

        # make sure the index are numpy array
        if isinstance(index, list):
            index = np.array(index, dtype=int)
        elif isinstance(index, int):
            index = np.array([index], dtype=int)

        # init the results
        res = {}
        nb_sample = index.size
        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)

        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el][:] = self.data[el][index, :]

        return res
