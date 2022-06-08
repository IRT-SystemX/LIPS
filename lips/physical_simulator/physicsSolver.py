# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking


class PhysicsSolver(object):
    """
    This class is the Base class that is used to create some "augmented simulator". These "augmented simulator" can be
    anything that emulates the behaviour of some "simulator".

    They are meant to use data coming from a `DataSet` to learn from it.
    """
    def __init__(self, name: str):
        self.name = name

        self._observations = dict()
        self._flow = dict()

    def compute(self, dataset: "DataSet"):
        """
        evaluate the model on the full dataset
        """
        pass

    def init(self, **kwargs):
        """
        initialize the "physics simulator".

        For example, this is where the model should be built in case the augmented simulator used a neural network.
        """
        pass

    def process_dataset(self, one_example):
        """
        This function transforms one state of a dataset (one row if you want) into something that can be used by
        the neural network (for example)
        """
        pass

    def data_to_dict(self):
        """
        This function should return two dictionaries in the following order
        - the observations used for evaluation
        - corresponding predictions
        """
        pass

    def save(self, path_out: str):
        """
        Ideally it should save the grid
        """
        pass

    def restore(self, path: str):
        """
        restore the grid from the path
        """
        pass
