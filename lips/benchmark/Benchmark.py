# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import json


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

        Simulator: An object of a sub class of ``Simulator`` class
            This is an object of Physical or Augmneted simulator including the prediction results

        evaluator: ``object`` of ``Evaluation`` class
            It allows to evaluate the performance of the simulator with respect to various point of views
            It should be parameterized before passing to benchmark class to include appropriate metrics 

        save_path : ``string``
            the root path of Outputs of the benchmark on the basis of which new directories are created

    """

    def __init__(self,
                 benchmark_name,
                 dataset,
                 simulator=None,
                 evaluator=None,
                 save_path=None,
                 ):

        self.benchmark_name = benchmark_name
        
        # object of class GenerateData contianing datasets for training and validation
        self.dataset = dataset
        # self.env = self.dataset.env
        # self.env_name = self.dataset.env_name

        # object of class Evaluation used to evaluate the benchmark and the model
        self.evaluator = evaluator
        self.metrics_ML = None
        self.metrics_physics = None
        self.metrics_adaptability = None
        self.metrics_readiness = None

        # object of class AugmentedSimulator allowing to train a neural net and to predict, load and save it 
        self.simulator = simulator

        self.observations, self.predictions = self.simulator.data_to_dict()
        self.prediction_time = self.simulator.predict_time

        
        """
        # attributes used for input and outputs of a model
        if self.augmentedSimulator:
            self.attr_x = self.augmentedSimulator.attr_x
            self.attr_y = self.augmentedSimulator.attr_y
            self.attr_tau = self.augmentedSimulator.attr_tau
        else:
            self.attr_x = attr_x
            self.attr_y = attr_y
            self.attr_tau = attr_tau

        self.attr_names = (*self.attr_x, *self.attr_tau, *self.attr_y)
        """

        # model and parameters
        # self.model_name = self.simulator.name


        # create directories
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        self.benchmark_path = os.path.join(save_path, benchmark_name)
        if not os.path.exists(self.benchmark_path):
            os.mkdir(self.benchmark_path)
    
    def evaluate_simulator(self,
                           choice="predictions",
                           EL_tolerance=0.04,
                           LCE_tolerance=1e-3,
                           KCL_tolerance=1e-2,
                           active_flow=True,
                           save_path=None):
        """
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object

        params
        ------
            choice: ``str``
                to compute physic compliances on predictions or on real observations 
                the choices are `predictions` or `observations`
            
            EL_tolerance: ``float``
                the tolerance used for electrical loss verification
            
            LCE_tolerance: ``float``
                the tolerance used for Law of Conservation of Energy

            KLC_tolerance: ``float``
                the tolerance used for Kirchhoff's current law

            active_flow: ``bool``
                whether to compute KCL on active (True) or reactive (False) powers

            save_path: ``str`` or ``None``
                if indicated the evaluation results will be saved to indicated path
        """

        self.evaluator.do_evaluations(env=None, 
                                      env_name=None,
                                      observations=self.observations, 
                                      predictions=self.predictions, 
                                      choice=choice,
                                      EL_tolerance=EL_tolerance,
                                      LCE_tolerance=LCE_tolerance,
                                      KCL_tolerance=KCL_tolerance,
                                      active_flow=active_flow,
                                      save_path=save_path)

        self.metrics_ML = self.evaluator.metrics_ML
        self.metrics_physics = self.evaluator.metrics_physics
        self.metrics_generalization = self.evaluator.metrics_generalization
        self.metrics_readiness = self.evaluator.metrics_readiness

    def save(self):
        """
        save the benchmark metadata and the trained model parameters for a further use

        It creates automatically a path using experiment name, benchmark name and model name to save the model
        """
        self._save_metadata(path=self.benchmark_path)

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
            self.simulator._save_metadata(path)
            self.simulator.save_data(path=path, ext=ext)
        
        elif self.simulator.model_path:
            self.simulator._save_metadata(self.simulator.model_path)
            self.simulator.save_data(path=self.simulator.model_path, ext=ext)
        
        else:
            print("the augmented simulator could not be saved")

    def _save_metadata(self, path=None):
        """
        Save the benchmark class metadata
        # TODO : save the training losses
        params
        ------
            path : ``str``
                the path where the benchmark metadata should be stored
        """
        res = self._get_metadata()
        json_nm = "metadata_Benchmark.json"
        with open(os.path.join(path, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=res, fp=f)

    def _get_metadata(self):
        """
        get the benchmark metadata in json serializable form
        TODO : save also the loss dicts : loss_metric_dict_train and loss_metric_dict_valid
        returns
        -------
            res : ``dict``
                a dictionary of benchmark metadata
        """
        res = dict()
        res["benchmark_name"] = self.benchmark_name
        res["benchmark_path"] = self.benchmark_path
        return res

    def load(self, path):
        """
        load the metadata for benchmark and load the model
        TODO : consider a special case for DC approximator
        """
        self._load_metadata(path)

    def _load_metadata(self, path=None):
        """
        load metadata for the benchmark
        TODO : consider a special case for DC approximator
        """
        json_nm = "metadata_Benchmark.json"
        with open(os.path.join(path, json_nm), "r", encoding="utf-8") as f:
            res = json.load(f)

        self.benchmark_name = res["benchmark_name"]
        self.benchmark_path = res["benchmark_path"]

    def visualize_network_state(self):
        """
        TODO : integrate the visualisation tools allowing to visualize the network state over different observations
        """
        pass