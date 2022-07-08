# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

#from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Union

from .utils import Mapper
from ..logger import CustomLogger
from ..config import ConfigManager
from ..dataset import DataSet
#from ..benchmark import Benchmark

class Evaluation(object):
    """Evaluation base class

    Evaluation class to be implemented for evaluation of an augmented simulator

    Hint
    ----
    Examples of such classes are provided in `PowerGridEvaluation` and `TransportEvaluation`

    Attributes
    ----------
    config : Union[ConfigManager, ``None``], optional
        the config object used for the evaluation, by default None
    config_path : Union[``str``, ``None``], optional
        the path where config file could be found,, by default None
    config_section : Union[``str``, ``None``], optional
        the section of config file to be used for evaluation, by default None
    log_path : Union[``str``, ``None``], optional
        the path where the logs should be stored or path to an existing log file, by default None
    """
    MACHINE_LEARNING = "ML"
    PHYSICS_COMPLIANCES = "Physics"
    INDUSTRIAL_READINESS = "IndRed"
    # OOD_GENERALIZATION = "OOD"

    def __init__(self,
                 config: Union[ConfigManager, None]=None,
                 config_path: Union[str, None]=None,
                 config_section: Union[str, None]=None,
                 log_path: Union[str, None]=None
                 ):
        if config is None:
            self.config = ConfigManager(section_name=config_section, path=config_path)
        else:
            self.config = config

        self.observations = dict()
        self.predictions = dict()
        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        self.mapper = Mapper()
        self.metrics = {}

    @classmethod
    #@abstractmethod
    def from_benchmark(cls,
                       benchmark: "Benchmark",
                       ):
        """
        Class method to intialize the evaluation from Benchmark instance

        Benchmark instance should include at least both `observations` and `predictions` variables
        """
        pass

    @classmethod
    def from_dataset(cls,
                     dataset: DataSet,
                     config_path: Union[str, None]=None,
                     config_section: Union[str, None]=None,
                     log_path: Union[str, None]=None):
        """
        Class method to initialize the evaluation from DataSet instance
        """
        pass

    def __init_metric_dict(self) -> dict:
        """
        Initialize the metrics dictionary structure

        It should be called if any modification to default category names
        """
        metrics_cat = {}
        metrics_cat[self.MACHINE_LEARNING] = {}
        metrics_cat[self.PHYSICS_COMPLIANCES] = {}
        metrics_cat[self.INDUSTRIAL_READINESS] = {}
        # metrics_cat[self.OOD_GENERALIZATION] = {}

        return metrics_cat

    def evaluate(self,
                 observations: dict,
                 predictions: dict,
                 save_path: Union[str, None]=None):
        """Evaluate the predictions of an AugmentedSimulator

        This function should be overridden to do all the required evaluations for each category

        - ML evaluation
        - Physics Compliance evaluation
        - Industrial Readiness evaluation
        - OOD Generalization evaluation

        Notes
        -----
        Notice that, the already minimalist code compute two metrics which are `MSE` and `MAE` from scikit-learn package

        Parameters
        ----------
        observations : ``dict``
            Observations used for evaluation of the augmented simulators
        predictions : ``dict``
            predictions obtained from augmented simulators
        save_path : Union[``str``, ``None``], optional
            path where the results should be saved, by default None, by default None
        """
        self.observations = observations #dataset.data
        self.predictions = predictions
        #self.logger.info("General metrics")
        generic_functions = self.mapper.map_generic_criteria()
        # create metrics dictionary
        self.metrics.update(self.__init_metric_dict())
        metric_dict = self.metrics[self.MACHINE_LEARNING]

        for metric_name, metric_fun in generic_functions.items():
            metric_dict[metric_name] = {}
            for nm_, pred_ in self.predictions.items():
                if nm_ == "__prod_p_dc":
                    # fix for the DC approximation
                    continue
                true_ = self.observations[nm_]
                try:
                    tmp = metric_fun(true_, pred_)
                except ValueError:
                    #Sklearn fails for 4D tensors, skip generic metric if error raised
                    continue
                if isinstance(tmp, Iterable):
                    metric_dict[metric_name][nm_] = [float(el) for el in tmp]
                    # self.logger.info("%s for %s: %s", metric_name, nm_, tmp)
                else:
                    metric_dict[metric_name][nm_] = float(tmp)
                    # self.logger.info("%s for %s: %.2f", metric_name, nm_, tmp)

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

        It taks multiple trained simulators and evaluates them on test datasets
        and finally it reports the results side by side
        """
        pass
