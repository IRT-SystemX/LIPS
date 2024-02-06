""" This module is used to generate pneumatic datasets

Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

import abc
from typing import Union
from collections.abc import Iterable

from lips.logger import CustomLogger
from lips.dataset.dataSet import DataSet
from lips.physical_simulator.physicalSimulator import PhysicalSimulator

class DataSetGeneratorBase(metaclass=abc.ABCMeta):
    def __init__(self,
                 name:str,
                 simulator:PhysicalSimulator,
                 attr_inputs:Iterable,
                 attr_outputs:Iterable,
                 attr_names:Iterable,
                 nb_samples:int,
                 log_path: Union[str, None]=None):

        self._name=name
        self._simulator=simulator
        self._attr_inputs=attr_inputs
        self._attr_outputs=attr_outputs
        self._attr_names=attr_names
        self._nb_samples=nb_samples
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        self._dataset_type=DataSet
        self._data=dict()

    @abc.abstractmethod
    def generate(self):
        pass

    @abc.abstractmethod
    def _init_data(self, simulator:PhysicalSimulator, nb_samples:int):
        pass

    @abc.abstractmethod
    def _generate_data(self):
        pass

    @abc.abstractmethod
    def _store_obs(self, current_size: int, obs:PhysicalSimulator):
        pass

    def _load_dataset_from_store_data(self):
        pass
        