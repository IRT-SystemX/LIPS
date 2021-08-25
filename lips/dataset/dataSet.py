# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from lips.physical_simulator import PhysicalSimulator


class DataSet(object):
    """
    This class represent a single dataset, that comes from a database (coming either from real world application
    or from generated data)

    If implemented it also offers the possibility to generate data. The data generation might come from a simulator
    that will be called when generating the dataset.

    This is the base class of all DataSet in LIPS repository
    """
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.data = None
        self.size = 0

    def __len__(self):
        return self.size

    def generate(self,
                 simulator,
                 actor,
                 path_out,
                 nb_samples,
                 simulator_seed=None,
                 actor_seed=None):
        """function to generate some data. It takes a "simulator" as input and will save the results into
        path_out.

        We expect that the results are saved in `path_out/simulation_name` if they are stored
        in plain text. Otherwise, if stored in a database we expect them to have (at least) some metadata involving the
        the `simulation_name`.

        Parameters
        ----------
        simulator:
            A simulator. The exact definition of the interface of the simulator is for now free. This means that a
            specific dataset class should be used for all different usecase.

        actor:
            Something that is able to modify the simulator to generate different kind of data. It can be ``None``
            if this is irrelevant.

        path_out:
            The path where the data should be saved

        nb_samples:
            The number of "step" to generate

        simulator_seed:
            Seed used to set the simulator for reproducible experiments

        actor_seed:
            Seed used to set the actor for reproducible experiments

        """
        assert isinstance(simulator, PhysicalSimulator), f"simulator should be a derived type of `PhysicalSimulator` " \
                                                         f"you provided {type(simulator)}"
        if actor is not None and not isinstance(actor, simulator.actor_types):
            raise RuntimeError(f"actor should be compatible with your simulator. You provided an actor of "
                               f"type {type(actor)} while your simulator accepts only actor from types "
                               f"{simulator.actor_types}")

    def load(self, path):
        """
        This function loads a dataset previously generated, for example by a call to `generate` it is expected
        to fail if it cannot match a dataset with the right experiment_name and "simulation_name"

        Parameters
        ----------
        path:
            The path to look for the dataset
        """
        pass

    def sample(self, nb_sample, sampler=None):
        """
        This functions samples uniformely at random some elements amongst the `self.data`

        If `nb_sample` is higher than the number of data in self.data then it samples with replacements.

        Parameters
        ----------
        nb_sample:
            Number of samples to sample from the dataset

        sampler:
            Currently unused, will be used to implement more complex sampling method

        Returns
        -------

        """
        if self.data is None:
            raise RuntimeError("Impossible to sample from a non initialized dataset. "
                               "Have you called `dataset.load(...)` "
                               "or `dataset.generate(...)` ?")

    def get_data(self, index):
        """
        This function returns the data in the data that match the index `index`

        Parameters
        ----------
        index:
            A list of integer

        Returns
        -------

        """
        if self.data is None:
            raise RuntimeError("Impossible to get_data from a non initialized dataset. "
                               "Have you called `dataset.load(...)` "
                               "or `dataset.generate(...)` ?")
