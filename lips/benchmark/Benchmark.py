# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import numpy as np
from tqdm import tqdm
import time
import copy
import shutil
import tempfile
import json
import importlib
from collections.abc import Iterable


from matplotlib import pyplot as plt

import tensorflow as tf

from lips.models import *

# for DC solver
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action import CompleteAction
from grid2op.Backend import PandaPowerBackend


class Benchmark():
    """
    Benchmark class which allow to build, train, evaluate different models on various power grid configurations corresponding to Benchmark 1

    params
    ------
        benchmark_name : ``str``
            a name attributed to the corresponding experiment

        generator : ``GenerateData`` object
            an object of GenerateData class which contains the data for training, validation and testing

        attr_x : ``tuple``
            the set of input variables to be used

        attr_y : ``tuple``
            the set of output variables which should be predicted

        attr_tau : ``tuple``
            the set of topology vectors that should be considered
    """

    def __init__(self,
                 benchmark_name,
                 dataset,
                 augmentedSimulator=None,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                         "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 attr_tau=("line_status",),
                 path="lips\\Outputs",
                 limit_gpu_memory=True
                 ):

        self.benchmark_name = benchmark_name
        
        # object of class GenerateData contianing datasets for training and validation
        self.dataset = dataset
        
        # object of class AugmentedSimulator allowing to train a neural net and to predict, load and save it 
        self.augmentedSimulator = augmentedSimulator

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

        # model and parameters
        self.model_name = None

        # predictions dictionary containing the predictions for each variable
        self.predictions = dict()
        self.observations = dict()

        # Training and evaluation time
        # TODO : to be removed and added directly to Evaluation object
        self.training_time = 0
        self.prediction_time = 0

        # creating a dictionary store train and validation losses and metrics
        self.loss_dicts_init = False
        self.loss_metric_dict_train = None
        self.loss_metric_dict_valid = None
        # the flag indicating if the model is trained
        self._model_trained = False

        # create directories
        benchmark_path = os.path.join(
            path, self.dataset.experiment_name, "Benchmarks")
        if not os.path.exists(benchmark_path):
            os.mkdir(benchmark_path)

        self.benchmark_path = os.path.join(benchmark_path, self.benchmark_name)
        if not os.path.exists(self.benchmark_path):
            os.mkdir(self.benchmark_path)

        self.model_path = None
        self.dc_path = None

        if limit_gpu_memory:
            self.limit_gpu_usage()

    def train_model(self, epochs=20, batch_size=32, shuffle=False, save_model=True, tf_writer=None, verbose=1):
        """
        Train the AugmentedSimulator._model using batches created from the DataSet.dataset["training"]

        Parameters
        -----------
            epochs: ``int``
                Value indicating the number of epochs required for the learning of the built model

            batch_size: ``int``
                Value indicating the batch size used during the learning phase

            save_model : ``bool``
                if True, save the model at self.benchmark_path

            tf_writer : ``tensorboard`` writer
                # TODO : not yet implemented

            verbose: ``bool``
                Whether to show or not the loss and accuracy information during training. The progress is managed by TQDM, it is advised to not use it.

            
        The steps are:
            1) take a batch using a function or not
            2) shuffle it or not
            3) transfer a batch to augmented simulator
            4) train the model on the batch
        """
        # verify if the model is built before training
        if self.augmentedSimulator and self.augmentedSimulator._model is None:
            raise RuntimeError("The model is not yet built !")
                
        # creating batches of data for training phase
        batches, steps = self.create_batches(tag="training", batch_size=batch_size)

        if shuffle:
            # TODO : shuffle the batches if required
            pass

        # whether to show the progress bar
        if verbose:
            pbar = tqdm(total=epochs*steps)

        # training and validation loop
        for epoch in range(epochs):
            train_losses_step = list()
            ################### training ##################
            # get batch should give a dictionary of all the available variables
            # a list of dictionary
            for step, batch in zip(np.arange(steps), batches):
                self.augmentedSimulator.store_batch(batch)
                train_batch_loss = self.augmentedSimulator.train(tf_writer=tf_writer)
                train_losses_step.append(train_batch_loss)

                if verbose:
                    pbar.update(1)
            if not(self.loss_dicts_init):
                self.init_loss_dicts()

            # compute the mean of losses per epoch for training
            mean_epoch_train = np.mean(np.vstack(train_losses_step), axis=0)
            for key_, val_ in zip(self.loss_metric_dict_train.keys(), mean_epoch_train):
                self.loss_metric_dict_train[key_].append(val_)
            if verbose == 2:
                print("Epoch number {} / {} with loss {}".format(epoch,
                      epochs, self.loss_metric_dict_train["loss"][-1]))

            # TODO : add a validations steps
        
        if verbose:
            pbar.close()

        self.model_trained = True
        self.model_name = self.augmentedSimulator.name
        self.training_time = self.augmentedSimulator._time_train


        # TODO : here we should call the save methods of proxy
        self.model_path = os.path.join(self.benchmark_path, self.model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        
        if save_model:
            self.save_augmentedSimulator(path=self.model_path)

    def init_loss_dicts(self):
        """
        initialize the losses dictionary from the built and trained model

        a loss value is associated to each output variable
        """
        # initialize loss dictionaries with losses from model
        self.loss_metric_dict_train = dict.fromkeys(self.augmentedSimulator._model.metrics_names)
        self.loss_metric_dict_valid = dict.fromkeys(self.augmentedSimulator._model.metrics_names)
        for key_ in self.augmentedSimulator._model.metrics_names:
            self.loss_metric_dict_train[key_] = list()
            self.loss_metric_dict_valid[key_] = list()
        self.loss_dicts_init = True

    def model_predict(self, batch_size=1024, tag="test", verbose=0, save_predictions=True, load_path=None):
        """
        this module will perform prediction on test sets, and the results could be sent to evaluation class to measure the performance
        of the experimented model

        params
        ------
            batch_size : ``int``
                inference batch size

            tag : ``str``
                the tag of a dataset on which inference should be performed using the trained model

            verbose : ``bool``
                whether to show the inference progress

            save_predictions : ``bool``
                if True save the predictions

            load_path : ``str``
                path where the fitted model is located, if it is not in memory

        """
        # TODO : maybe to verify if the model is already trained
        if (self.augmentedSimulator._model is None) & (load_path is None):
            raise RuntimeError(
                "The model is not in memory nor a path is indicated")
        elif load_path:
            self.augmentedSimulator.load(load_path)
            self.model_path = load_path
        elif self.model_path:
            self.augmentedSimulator.load(self.model_path)

        # a key for each tag and each variable
        self.predictions[tag] = {}
        for _var_nm in self.attr_y:
            self.predictions[tag][_var_nm] = list()

        # creating batches of data
        batches, steps = self.create_batches(tag=tag, batch_size=batch_size)

        if verbose:
            pbar = tqdm(total=steps)

        # predicting over batches
        for step, batch in zip(np.arange(steps), batches):
            self.augmentedSimulator.store_batch(batch)
            res = self.augmentedSimulator.predict()

            for _var_nm, _pred in zip(self.attr_y, res):
                self.predictions[tag][_var_nm].append(_pred)
            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        self.prediction_time = self.augmentedSimulator._time_predict


        # keeping the prediction in array format for evaluation
        for _var_nm in self.attr_y:
            self.predictions[tag][_var_nm] = np.asarray(
                np.vstack(self.predictions[tag][_var_nm]), dtype=np.float32)

        # a deep copy of observation which are the ground truth values for predictions
        self.observations[tag] = copy.deepcopy(
            self.dataset.dataset[tag])

        # TODO : save the predictions
        self.model_path = os.path.join(self.benchmark_path, self.augmentedSimulator.name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if save_predictions:
            self.save_predictions(save_path=self.model_path)


    def evaluate_dc(self,
                    name="dc_approx",
                    is_dc=True,
                    # input that will be given to the proxy
                    attr_x=("prod_p", "prod_v", "load_p",
                            "load_q", "topo_vect"),
                    # output that we want the proxy to predict
                    attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                            "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                    tag="test",
                    save_evaluation=True,
                    verbose=1
                    ):
        """
        Evaluate the DC approximation which is used as a baseline for the comparison with other models.

        params
        ------
            name : ``str``
                a name for the DC solver

            is_dc : ``bool``
                whether the solver use DC approximation

            attr_x : ``tuple``
                the set of input variables used for DC approximator, topology vector should be appreard in this tuple

            attr_y : ``tuple``
                the set of output variables used for DC approximator. 

            tag : ``str``
                which dataset should be used for evaluation using DC approximator

            save_evaluation : ``bool``
                if True save the DC predictions

            verbose : ``bool``
                if true, the progress bar will be visualized
        """
        self.model_name = "DC"

        self.predictions[tag] = {}
        nb_observations = self.dataset.dataset_size[tag]

        for _var_nm in attr_y:
            self.predictions[tag][_var_nm] = list()

        _sz_x = [self.dataset.dataset_original[tag].get(
            nm_).shape[1] for nm_ in attr_x]
        #_sz_y = [self.generator.dataset_original[tag].get(nm_).shape[1] for nm_ in attr_y]

        input_array = []
        for sz in _sz_x:
            input_array.append(
                np.zeros((nb_observations, sz), dtype=np.float32))

        for attr_nm, inp in zip(attr_x, input_array):
            inp[:, :] = self.dataset.dataset_original[tag].get(attr_nm)

        # init the solver
        solver, _bk_act_class, _act_class, _indx_var = self._init_dc_solver(self.dataset.env._init_grid_path,
                                                                            name,
                                                                            attr_x,
                                                                            attr_y)

        # iterate over test set to infer the output variables using DC approximation
        if verbose:
            pbar = tqdm(total=nb_observations)

        for idx in range(nb_observations):
            res, _pred_time = DCApproximation(solver=solver,
                                              _bk_act_class=_bk_act_class,
                                              _act_class=_act_class,
                                              _indx_var=_indx_var,
                                              idx=idx,
                                              input_array=input_array,
                                              is_dc=is_dc,
                                              attr_x=attr_x,
                                              attr_y=attr_y
                                              )

            self.prediction_time += _pred_time

            for _var_nm, _pred in zip(attr_y, res):
                self.predictions[tag][_var_nm].append(_pred)

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        for _var_nm in self.attr_y:
            self.predictions[tag][_var_nm] = np.asarray(
                np.vstack(self.predictions[tag][_var_nm]), dtype=np.float32)

        self.observations[tag] = copy.deepcopy(
            self.dataset.dataset_original[tag])

        if save_evaluation:
            self.dc_path = os.path.join(self.benchmark_path, "DC")
            if not os.path.exists(self.dc_path):
                os.mkdir(self.dc_path)
            self.save_predictions(save_path=self.dc_path)

    def _init_dc_solver(self, path_grid_json, name, attr_x, attr_y):
        """
        initialize the dc solver which is used afterwards to predict the desired variables using inputs

        params
        ------
            path_grid_json : ``str``
                the path to the Grid2op environment

            name : ``str``
                a name for the DC sovler

            attr_x : ``tuple``
                the set of input variables used for DC approximator, topology vector should be appreard in this tuple

            attr_y : ``tuple``
                the set of output variables used for DC approximator.

        Returns
        --------
            solver : ``PandaPowerBackend``
                the DC solver

            _bk_act_class and _act_class

            _indx_var : ``dict``
                a dictionary comprising the indices of attr_x
        """
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            if not el in attr_x:
                raise RuntimeError(
                    f"The DC approximation need the variable \"{el}\" to be computed.")

        _supported_output = {"a_or", "a_ex", "p_or", "p_ex",
                             "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"}
        for el in attr_y:
            if not el in _supported_output:
                raise RuntimeError(f"This solver cannot output the variable \"{el}\" at the moment. "
                                   f"Only possible outputs are \"{_supported_output}\".")

        solver = PandaPowerBackend()
        solver.set_env_name(name)
        solver.load_grid(path_grid_json)
        solver.assert_grid_correct()
        _bk_act_class = _BackendAction.init_grid(solver)
        _act_class = CompleteAction.init_grid(solver)

        _indx_var = {}
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            _indx_var[el] = attr_x.index(el)

        return solver, _bk_act_class, _act_class, _indx_var

    def create_batches(self, tag="training", batch_size=32):
        """
        create batches of required data

        function create batches of data which could be used for learning and prediction steps of the model

        params
        ------
            tag : ``str``
                indicate the tag of the desired dataset for which we wish have the batches

            batch_size : ``int``
                the batch sizes which indicates how many samples should the model treate at each step of learning

        returns
        -------
            batches : ``list``
                a list of batches which are dictionaries of all the available variables

            steps : ``int``
                indicates number of batches which should be introduced to the model
        """
        steps = np.int(
            np.ceil((self.dataset.dataset_size[tag]) / batch_size))

        batches = list()

        for step in range(steps-1):
            tmp_dict = dict()
            for nm_ in self.attr_names:
                tmp_dict[nm_] = self.dataset.dataset[tag].get(nm_)[step*batch_size: (step+1)*batch_size]
            batches.append(tmp_dict)

        tmp_dict = dict()
        for nm_ in self.attr_names:
            tmp_dict[nm_] = self.dataset.dataset[tag].get(nm_)[(step+1)*batch_size:]
        batches.append(tmp_dict)

        return batches, steps

    def save(self):
        """
        save the benchmark metadata and the trained model parameters for a further use

        It creates automatically a path using experiment name, benchmark name and model name to save the model
        """
        self._save_metadata(path=self.benchmark_path)

    
    def save_augmentedSimulator(self, path=None, ext=".h5"):
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
            self.augmentedSimulator._save_metadata(path)
            self.augmentedSimulator.save_data(path=path, ext=ext)
        
        elif self.model_path:
            self.augmentedSimulator._save_metadata(self.model_path)
            self.augmentedSimulator.save_data(path=self.model_path, ext=ext)
        
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
        res["model_name"] = self.model_name
        res["attr_x"] = self.attr_x
        res["attr_y"] = self.attr_y
        res["attr_tau"] = self.attr_tau
        res["benchmark_path"] = self.benchmark_path
        res["model_path"] = self.model_path
        res["dc_path"] = self.dc_path
        res["train_losses"] = self.loss_metric_dict_train
        res["valid_losses"] = self.loss_metric_dict_valid

        return res


    def load(self, path):
        """
        load the metadata for benchmark and load the model
        TODO : consider a special case for DC approximator
        """
        self._load_metadata(path)

        for tag in self.dataset.tag_list:
            self.observations[tag] = copy.deepcopy(
                self.dataset.dataset[tag])

    def _load_metadata(self, path=None):
        """
        load metadata for the benchmark
        TODO : consider a special case for DC approximator
        """
        json_nm = "metadata_Benchmark.json"
        with open(os.path.join(path, json_nm), "r", encoding="utf-8") as f:
            res = json.load(f)

        self.benchmark_name = res["benchmark_name"]
        self.model_name = res["model_name"]
        self.attr_x = res["attr_x"]
        self.attr_y = res["attr_y"]
        self.attr_tau = res["attr_tau"]
        self.loss_metric_dict_train = res["train_losses"]
        self.loss_metric_dict_valid = res["valid_losses"]
        self.benchmark_path = res["benchmark_path"]
        self.model_path = res["model_path"]
        self.dc_path = res["dc_path"]

    def _save_dict(self, li, val):
        """
        save the metadata of generator into a valid representation of `self`. It's a utility function to convert
        into "json serializable object" the numpy data.

        It is a helper to convert data in float format from either a list or a single numpy floating point.

        Parameters
        ----------
        li
        val

        """
        if isinstance(val, Iterable):
            li.append([float(el) for el in val])
        else:
            li.append(float(val))

    def save_predictions(self, save_path=None):
        """
        save the model predictions in file
        """

        # save the data for each existing tag
        for key_, val_ in self.predictions.items():
            tag_dir = os.path.join(save_path, key_)
            if not os.path.exists(tag_dir):
                os.mkdir(tag_dir)
            for key__, val__ in val_.items():
                np.save(os.path.join(tag_dir, f"{key__}_pred.npy"), val__)
        print("Predictions saved successfully !")

    def load_predictions(self, load_path=None, tag=None):
        """
        Load the predictions and corresponding observations
        TODO : to be implemented
        """
        if self.dc_path:
            pass

    def limit_gpu_usage(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        # Currently, memory growth needs to be the same across GPUs
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def plot_convergence(self):
        """
        Visualizing the convergence of the model using training and validation losses curve in function of epochs

        """
        if self.loss_metric_dict_train:
            epochs = len(self.loss_metric_dict_train["loss"])
            # plot here if the loss metric dict is populated
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.loss_metric_dict_train["loss"], label="training loss")
            plt.plot(
                self.loss_metric_dict_valid["loss"], label="validation loss")
            plt.legend()
            plt.grid()
            plt.ylabel(f"{self.augmentedSimulator.loss} value")
            plt.xlabel("Epochs")
            plt.xticks(np.arange(epochs), np.arange(epochs)+1)
            plt.show()

    def visualize_network_state(self):
        """
        TODO : integrate the visualisation tools allowing to visualize the network state over different observations
        """
        pass