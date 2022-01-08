# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
from typing import Union
import numpy as np

from lips.evaluation import Evaluation
from lips.augmented_simulators import AugmentedSimulator
from lips.physical_simulator import PhysicsSolver

class Benchmark(object):
    """
    Benchmark class that takes a test dataset, a simulator (physical or augmented) and an evaluator object and 
    evaluates the simulator on test dataset with respect to required metrics requested by evaluator

    params
    ------
        benchmark_name : ``str``
            a name attributed to the corresponding experiment

        dataset: ``DataSet`` object
            an object of DataSet class which containing the data for training, validation and testing

        Simulator: An object from an augmented simulator
            This is an object of Physical or Augmneted simulator including the prediction results

        evaluation: ``object`` of ``Evaluation`` class
            It allows to evaluate the performance of the simulator with respect to various point of views
            It should be parameterized before passing to benchmark class to include appropriate metrics
            otherwise, it is initialized using an empty dictionary 

        path_benchmark : ``string``
            the path used to save the benchmark results

    """
    def __init__(self,
                 benchmark_name: str,
                 dataset: Union["DataSet", None]=None,
                 augmented_simulator: Union[AugmentedSimulator, PhysicsSolver, None]=None,
                 evaluation: Union[Evaluation, None]=None,
                 path_benchmark: Union[str, None]=None,
                 log_path: Union[str, None]=None
                 ):
        self.benchmark_name = benchmark_name
        self.path_benchmark = path_benchmark
        self.path_datasets = os.path.join(path_benchmark, self.benchmark_name)

        # Object of class DataSet contianing datasets for testing
        # It contains the last dataset used for evaluation
        self.dataset = dataset

        # Store the test data sets and simulator predictions for further investigations
        # Each evaluated dataset added to the dictionary (key=dataset.name)
        self.observations = dict()
        self.predictions = dict()

        # object of class Evaluation used to evaluate the augmented simulator on test data
        if evaluation is None:
            self.evaluation = Evaluation(log_path=log_path)
            # initialize it with empty dictionary, to be modified for a desired benchmark
            self.evaluation.set_active_dict(self.evaluation.get_empty_active_dict())
        else:
            # otherwise, use the evaluation object with initialized dictionary values
            self.evaluation = evaluation

        # object of class AugmentedSimulator allowing to train a neural net and to predict, load and save it 
        self.augmented_simulator = augmented_simulator

        
        # create directories for benchmark TODO : not necessary, to be removed
        #if path_benchmark is not None:
        #    if not os.path.exists(path_benchmark):
        #        os.mkdir(path_benchmark)

        #if not os.path.exists(self.path_datasets):
        #    os.mkdir(self.benchmark_path)
    
    def evaluate_simulator(self,
                           dataset,
                           augmented_simulator,
                           batch_size=32,
                           save_path=None):
        """
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be overloaded or called to evaluate the performance on multiple datasets
        
        params
        ------
            dataset: ``DataSet`` object
                a test dataset on which the augmented simulator should be performed and evaluated by

            augmented_simulator: ``AugmentedSimulator`` object 
                a trained augmented simulator which should be evaluated

            batch_size: ``int``
                evaluation batch size

            save_path: ``str`` or ``None``
                if indicated the evaluation results will be saved to indicated path
        """
        self.augmented_simulator = augmented_simulator
        self.dataset = dataset
        predictions = self.augmented_simulator.evaluate(dataset, batch_size)
        observations = self.dataset.get_data(np.arange(len(dataset)))
        # TODO : the ``do_evaluation`` function should have a more general interface in future
        # TODO : the ``Evaluation`` module should be refactored in near future
        res = self.evaluation.do_evaluations(predictions=predictions,
                                             observations=observations,
                                             choice="predictions",  # we want to evaluate only the predictions here
                                             save_path=save_path  # TODO currently not used
                                             )
        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = observations

        return res

    def __save_augmentedSimulator(self, path=None, ext=".h5"):
        """
        save the augmented simulator related data on model_path

        the data concerns the meta data and model itself

        !! we recommend to save to the default path for a better organisation (keep path=None)

        params
        ------
            path: ``str``
                if not indicated, the data will be saved in predefined path
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