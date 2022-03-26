# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Union

from .utils import Mapper
from ..logger import CustomLogger
from ..config import ConfigManager

class Evaluation(ABC):
    """
    Evaluation class to be implemented for evaluation of an augmented simulator

    Examples of such classes are provided in `PowerGridEvaluation` and `TransportEvaluation`

    Functions highlighted by @abstractmethod should be implemented

    Parameters
    ==========
    observations: `dict`
        a dictionary of observations including the key value pairs for test dataset observations

    predictions: `dict`
        a dictionary of predictions made by an augmented simulator

    config: `ConfigParser` section
        a section of ConfigParser class giving some parameters for evaluation (thresholds, etc.)

    log_path: `str`
        the path where the logs should be stored or path to an existing log file
    """
    MACHINE_LEARNING = "ML"
    PHYSICS_COMPLIANCES = "Physics"
    INDUSTRIAL_READINESS = "IndRed"
    OOD_GENERALIZATION = "OOD"

    def __init__(self,
                 observations: Union[dict, None]=None,
                 predictions: Union[dict, None]=None,
                 config_path: Union[str, None]=None,
                 config_section: Union[str, None]=None,
                 log_path: Union[str, None]=None):
        # generic init class to be able evaluate external results
        self.observations = observations
        self.predictions = predictions
        self.config_path = config_path
        self.config_section = config_section
        self.config = ConfigManager(self.config_section, path=self.config_path)
        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        self.mapper = Mapper()
        self.metrics = None
        self._init_metric_dict()

    @classmethod
    @abstractmethod
    def from_benchmark(cls, benchmark, config_path, config_section, log_path):
        """
        Class method to intialize the evaluation from Benchmark instance

        Benchmark instance should include at least both `observations` and `predictions` variables
        """
        pass

    @classmethod
    def from_dataset(cls, dataset, config_path, config_section, log_path: Union[str, None]=None):
        """
        Class method to initialize the evaluation from DataSet instance
        """
        pass

    def _init_metric_dict(self) -> dict:
        """
        Initialize the metrics dictionary structure

        It should be called if any modification to default category names
        """
        self.metrics = {}
        self.metrics[self.MACHINE_LEARNING] = {}
        self.metrics[self.PHYSICS_COMPLIANCES] = {}
        self.metrics[self.INDUSTRIAL_READINESS] = {}
        self.metrics[self.OOD_GENERALIZATION] = {}

        return self.metrics

    def evaluate(self, save_path: Union[str, None]=None) -> dict:
        """
        This function should be overridden to do all the required evaluations for each category

        - ML evaluation
        - Physics Compliance evaluation
        - Industrial Readiness evaluation
        - OOD Generalization evaluation

        The child classes should override this class for further extensions
        - PowerGridEvaluation
        - TransportEvaluation

        > Notice that, the already minimalist code compute two metrics which are `MSE` and `MAE` from scikit-learn package
        
        """
        self.logger.info("General metrics")
        generic_functions = self.mapper.map_generic_criteria()
        metric_dict = self.metrics[self.MACHINE_LEARNING]
        
        for metric_name, metric_fun in generic_functions.items():
            metric_dict[metric_name] = {}
            for nm_, pred_ in self.predictions.items():
                if nm_ == "__prod_p_dc":
                    # fix for the DC approximation
                    continue
                true_ = self.observations[nm_]
                tmp = metric_fun(true_, pred_)
                if isinstance(tmp, Iterable):
                    metric_dict[metric_name][nm_] = [float(el) for el in tmp]
                    self.logger.info("%s for %s: %s", metric_name, nm_, tmp)
                else:
                    metric_dict[metric_name][nm_] = float(tmp)
                    self.logger.info("%s for %s: %.2f", metric_name, nm_, tmp)

        # TODO : don't forget to save the results
        if save_path:
            pass

    def evaluate_ml(self):
        """
        It evaluates machine learning specific criteria
        """
        pass
    
    def evaluate_physics(self):
        """
        It should evaluate Physics Compliances 
        """
        pass

    def evaluate_industrial_readiness(self):
        """
        It should evaluate the industrial readiness of augmented simulators
        """
        pass
    
    def evaluate_ood(self):
        """
        It should evaluate out-of-distribution capacity of augmented simulators
        """
        pass

    def compare_simulators(self):
        """
        Bonus function

        It taks multiple trained simulators and evaluates them on test datasets and finally it reports the results side by side
        """
        pass





