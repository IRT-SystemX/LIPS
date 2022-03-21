"""
Usage:
    Helper classes and functions for evaluation

License:
    Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0

Description:
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""
import copy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class Mapper(object):
    """
    This class create a mapping dict between evaluation criteria and their respective functions available in metrics module

    TODO:
        - it should register the mapping meta data to a file
        - to be able reload it from file
        - to be able to update it (modify current and add new mappings)

    Parameters
    ==========

    """
    def __init__(self):
        self.criteria = {}
        self.map_generic_criteria()

    def map_generic_criteria(self):
        """
        Get a list of generic evaluation functions
        """
        tmp_criteria = {}
        tmp_criteria.update(MSE=mean_squared_error)
        tmp_criteria.update(MAE=mean_absolute_error)
        self.criteria.update(copy.deepcopy(tmp_criteria))
        
        return copy.deepcopy(tmp_criteria)

    def map_powergrid_criteria(self):
        """
        It populates the dictionary with available powergrids critiera
        """
        from ..metrics.power_grid import physics_compliances
        from ..metrics.power_grid import DEFAULT_METRICS
        self.criteria.update(DEFAULT_METRICS)
        self.criteria.update(CURRENT_POS=physics_compliances.verify_current_pos)
        self.criteria.update(VOLTAGE_POS=physics_compliances.verify_voltage_pos)
        self.criteria.update(LOSS_POS=physics_compliances.verify_loss_pos)
        self.criteria.update(DISC_LINES=physics_compliances.verify_disc_lines)
        self.criteria.update(CURRENT_EQ=physics_compliances.verify_current_eq)
        self.criteria.update(CHECK_LOSS=physics_compliances.verify_loss)
        self.criteria.update(LOSS_POS=physics_compliances.verify_energy_conservation)
        self.criteria.update(LOSS_POS=physics_compliances.verify_kcl)

        return self.criteria

    def map_label_to_func(self, label: str, func):
        """
        create new mappings
        """
        self.criteria[label] = func

    def get_func(self, label: str):
        """
        get the function for a metric
        """
        return self.criteria.get(label)

    def rename_key(self, k_old, k_new):
        """
        rename the a dictionary key
        """
        self.criteria[k_new] = self.criteria[k_old]

