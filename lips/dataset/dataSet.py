"""
Usage:
    DataSet is a base class for creating datasets
Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""
from typing import Union, Callable

from lips.physical_simulator import PhysicalSimulator

class DataSet(object):
    """Base class for data management
    This class represent a single dataset, that comes from a database (coming either from real world application
    or from generated data)

    If implemented it also offers the possibility to generate data. The data generation might come from a simulator
    that will be called when generating the dataset.

    This is the base class of all DataSet in LIPS repository

    Parameters
    ----------
    name: str
        the name of the dataset
    """
    def __init__(self, name:str):
        self.name = name
        self.data = None
        self.size = 0
        self.current_index = 0

    def generate(self,
                 simulator: PhysicalSimulator,
                 actor: Union[None, object],
                 path_out: Union[str, None],
                 nb_samples: int,
                 simulator_seed: Union[None, int]=None,
                 actor_seed: Union[None, int]=None):
        """Generate a dataset from a simulator
        function to generate some data. It takes a "simulator" as input and will save the results into path_out.

        We expect that the results are saved in `path_out/simulation_name` if they are stored
        in plain text. Otherwise, if stored in a database we expect them to have (at least) some metadata involving the
        the `simulation_name`.

        Parameters
        ----------
        simulator: PhysicalSimulator
            A simulator. The exact definition of the interface of the simulator is for now free. This means that a
            specific dataset class should be used for all different usecase.
        actor: Union[None, object]
            Something that is able to modify the simulator to generate different kind of data. It can be ``None``
            if this is irrelevant.
        path_out: Union[str, None]
            The path where the data should be saved
        nb_samples: int
            The number of "step" to generate
        simulator_seed: Union[None, int]
            Seed used to set the simulator for reproducible experiments
        actor_seed: Union[None, int]
            Seed used to set the actor for reproducible experiments
        """
        assert isinstance(simulator, PhysicalSimulator), f"simulator should be a derived type of `PhysicalSimulator` " \
                                                         f"you provided {type(simulator)}"
        if actor is not None and not isinstance(actor, simulator.actor_types):
            raise RuntimeError(f"actor should be compatible with your simulator. You provided an actor of "
                               f"type {type(actor)} while your simulator accepts only actor from types "
                               f"{simulator.actor_types}")

    def load(self, path:str):
        """Load a dataset from a file
        This function loads a dataset previously generated, for example by a call to `generate` it is expected
        to fail if it cannot match a dataset with the right experiment_name and "simulation_name"

        Parameters
        ----------
        path: str
            The path to look for the dataset
        """
        pass

    def sample(self, nb_sample: int, sampler: Callable=None) -> dict:
        """Sample a dataset

        This functions samples uniformely at random some elements amongst the `self.data`
        If `nb_sample` is higher than the number of data in self.data then it samples with replacements.

        Parameters
        ----------
        nb_sample : int
            Number of samples to sample from the dataset
        sampler : Callable, optional
            Currently unused, will be used to implement more complex sampling method, by default None

        Returns
        -------
        dict
            sample of data

        Raises
        ------
        RuntimeError
            Impossible to sample from a non initialized dataset

        """
        if self.data is None:
            raise RuntimeError("Impossible to sample from a non initialized dataset. "
                               "Have you called `dataset.load(...)` "
                               "or `dataset.generate(...)` ?")

    def get_data(self, index: tuple):
        """get the data at a specific index

        This function returns the data in the data that match the index `index`

        Parameters
        ----------
        index : tuple
            A tuple of integer

        """
        if self.data is None:
            raise RuntimeError("Impossible to get_data from a non initialized dataset. "
                               "Have you called `dataset.load(...)` "
                               "or `dataset.generate(...)` ?")

    def __len__(self):
        return self.size
    
    # def __len__(self)->int:
    #     data = self.get_data()
    #     return data[list(data.keys())[0]].shape[0]

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self):
            current_data = self.__getitem__(self.current_index)
            self.current_index += 1
            return current_data
        raise StopIteration

    def __getitem__(self, index:int):
        return self.get_data(index)
