# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
from typing import Union
from abc import ABC, abstractmethod

import pathlib
import numpy as np

from ..evaluation import Evaluation
from ..augmented_simulators import AugmentedSimulator
from ..physical_simulator import PhysicalSimulator
from ..physical_simulator import PhysicsSolver
from ..config import ConfigManager
from ..logger import CustomLogger
from ..dataset import DataSet


class Benchmark(ABC):
    """Benchmark Base Class

    Benchmark class that takes a test dataset, a simulator (physical or augmented) and an
    evaluator object and evaluates the simulator on test dataset with respect to required
    metrics requested by evaluator

    Attributes
    ----------
    benchmark_name : ``str``
        a name attributed to the corresponding experiment
    benchmark_path : Union[``pathlib.Path``, ``str``, ``None``], optional
        the path to benchmark scenarios and related datasets
    config_path : Union[``pathlib.Path``, ``str``]
        the path of config file
    dataset : Union[``DataSet``, ``None``], optional
        an object of ``DataSet`` class which containing the data for training, validation and testing, by default None
    augmented_simulator : Union[``AugmentedSimulator``, ``PhysicsSolver``, ``None``], optional
        This is an object of Physical or Augmneted simulator including the prediction results, by default None
    evaluation : Evaluation, optional
        It allows to evaluate the performance of the simulator with respect to various point of views
        It should be parameterized before passing to benchmark class to include appropriate metrics
        otherwise, it is initialized using an empty dictionary, by default None
    log_path : Union[``pathlib.Path``, ``str``, ``None``], optional
        the path of logger, by default None
    """
    def __init__(self,
                 benchmark_name: str,
                 benchmark_path: Union[pathlib.Path, str, None],
                 config_path: Union[pathlib.Path, str],
                 dataset: Union[DataSet, None]=None,
                 augmented_simulator: Union[AugmentedSimulator, PhysicsSolver, None]=None,
                 evaluation: Evaluation=None,
                 log_path: Union[pathlib.Path, str, None]=None,
                 **kwargs
                 ):
        self.benchmark_name = benchmark_name
        self.benchmark_path = benchmark_path
        self.path_datasets = os.path.join(benchmark_path, self.benchmark_name) if benchmark_path else None

        # config file
        if not(os.path.exists(config_path)):
            raise RuntimeError("Configuration path not found for the benchmark!")
        elif not str(config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        else:
            self.config = ConfigManager(section_name=benchmark_name, path=config_path)
        self.config.set_options_from_dict(**kwargs)
        # Object of class DataSet contianing datasets for testing
        # It contains the last dataset used for evaluation
        self.dataset = dataset

        # Store the test data sets and simulator predictions for further investigations
        # Each evaluated dataset added to the dictionary (`key` corresponds to `dataset.name`)
        self.observations = dict()
        self.predictions = dict()

        # object of class AugmentedSimulator allowing to train a neural net and to predict, load and save it
        self.augmented_simulator = augmented_simulator

        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger

        if evaluation is not None:
            # use the user-defined evaluation object
            self.evaluation = evaluation

    @abstractmethod
    def evaluate_simulator(self,
                           dataset: DataSet,
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           **kwargs) -> dict:
        """
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be overloaded or called to evaluate the performance on multiple datasets

        Parameters
        ----------
        dataset: DataSet
            a test dataset on which the augmented simulator should be performed and evaluated by
        augmented_simulator: AugmentedSimulator
            a trained augmented simulator which should be evaluated
        save_path: Union[``str``, ``None``]
            if indicated the evaluation results will be saved to indicated path
        **kwargs: ``dict``
            additional parameters to be passed to the evaluator for augmented simulator
        Returns
        -------
        ``dict``
            a dictionary containing the evaluation results
        """
        self.augmented_simulator = augmented_simulator
        self.dataset = dataset
        predictions = self.augmented_simulator.evaluate(dataset, **kwargs)
        observations = self.dataset.get_data(np.arange(len(dataset)))
        res = self.evaluation.evaluate(observations=observations,
                                       predictions=predictions,
                                       save_path=save_path
                                      )
        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = observations

        return res

    def __save_augmented_simulator(self, path=None, ext=".h5"):
        """
        save the augmented simulator related data on model_path

        the data concerns the meta data and model itself

        !! we recommend to save to the default path for a better organisation (keep path=None)

        Parameters
        ------
        path: ``str``
            if not indicated, the data will be saved in predefined path
        ext: ``str``
            extension
        """
        if path:
            if not os.path.exists(path):
                os.mkdir(path)
            self.augmented_simulator._save_metadata(path)
            self.augmented_simulator.save_data(path=path, ext=ext)

        elif self.augmented_simulator.model_path:
            self.augmented_simulator._save_metadata(self.augmented_simulator.model_path)
            self.augmented_simulator.save_data(path=self.augmented_simulator.model_path, ext=ext)

        else:
            print("the augmented simulator could not be saved")