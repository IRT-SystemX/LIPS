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
from typing import Callable
import copy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np
#sklearn does not support 4D tensors
def mean_absolute_error2(sols,preds): 
    sols,preds=np.array(sols),np.array(preds)
    absError = np.abs(preds - sols)
    return np.mean(absError)

def mean_squared_error2(sols,preds): 
    sols,preds=np.array(sols),np.array(preds)
    squaredError = (preds - sols)**2
    return np.mean(squaredError)


class Mapper(object):
    """Mapper class
    This class create a mapping dict between evaluation criteria and their respective functions available in metrics module

    TODO:
    - it should register the mapping meta data to a file
    - to be able reload it from file
    - to be able to update it (modify current and add new mappings)

    """
    def __init__(self):
        self.criteria = {}
        self.map_generic_criteria()

    def map_generic_criteria(self):
        """
        Get a list of generic evaluation functions

        """
        tmp_criteria = {}
        tmp_criteria.update(MSE_avg=mean_squared_error)
        tmp_criteria.update(MAE_avg=mean_absolute_error)
        tmp_criteria.update(MSE_avg2=mean_squared_error2)
        tmp_criteria.update(MAE_avg2=mean_absolute_error2)
        self.criteria.update(copy.deepcopy(tmp_criteria))

        return copy.deepcopy(tmp_criteria)

    def map_powergrid_criteria(self):
        """
        It populates the dictionary with available powergrids critiera

        """
        from lips.metrics.power_grid import physics_compliances
        from lips.metrics.power_grid.verify_voltage_equality import verify_voltage_at_bus
        from lips.metrics.power_grid.local_conservation import local_conservation
        from lips.metrics.power_grid.global_conservation import global_conservation
        from lips.metrics import DEFAULT_METRICS
        self.criteria.update(DEFAULT_METRICS)
        self.criteria.update(CURRENT_POS=physics_compliances.verify_current_pos)
        self.criteria.update(VOLTAGE_POS=physics_compliances.verify_voltage_pos)
        self.criteria.update(LOSS_POS=physics_compliances.verify_loss_pos)
        self.criteria.update(DISC_LINES=physics_compliances.verify_disc_lines)
        self.criteria.update(CURRENT_EQ=physics_compliances.verify_current_eq)
        self.criteria.update(CHECK_LOSS=physics_compliances.verify_loss)
        self.criteria.update(CHECK_GC=global_conservation)
        self.criteria.update(CHECK_LC=local_conservation)
        self.criteria.update(CHECK_KCL=physics_compliances.verify_kcl)
        self.criteria.update(CHECK_VOLTAGE_EQ=verify_voltage_at_bus)

        return self.criteria

    def map_label_to_func(self, label: str, func: Callable):
        """create new mappings

        Parameters
        ----------
        label : str
            a new label
        func : Callable
            a metric function
        """
        self.criteria[label] = func

    def get_func(self, label: str) -> Callable:
        """get the function for a metric

        Parameters
        ----------
        label : str
            the label of the metric

        Returns
        -------
        Callable
            the corresponding function
        """
        return self.criteria.get(label)

    def rename_key(self, k_old: str, k_new: str):
        """rename the a dictionary key

        Parameters
        ----------
        k_old : str
            old key
        k_new : str
            new key
        """
        self.criteria[k_new] = self.criteria[k_old]

