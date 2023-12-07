# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import pathlib
from typing import Union
import shutil
import json
import tempfile
import importlib

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from . import AugmentedSimulator
from ..utils import NpEncoder
from ..dataset import DataSet
from ..logger import CustomLogger

class TensorflowSimulator(AugmentedSimulator):
    """_summary_

        Parameters
        ----------
        name : str, optional
            _description_, by default None
        config : ConfigManager
            _description_
        """
    def __init__(self,
                 name: Union[str, None]=None,
                 log_path: Union[str, None] = None,
                 **kwargs):
        super().__init__(name=name, log_path=log_path, **kwargs)
        # logger
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        self._optimizer = None

        self.input_size = None
        self.output_size = None

        # setting seeds
        np.random.seed(1)
        tf.random.set_seed(2)


    def build_model(self):
        """build tensorflow model

        Parameters
        ----------
        **kwargs : dict
            if parameters indicated, it will replace config parameters

        Returns
        -------
        keras.Model
            _description_
        """
        if self.input_size is None or self.output_size is None:
            raise RuntimeError("input_size is not set")


    def train(self,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet] = None,
              save_path: Union[None, str] = None,
              **kwargs):
        """Function used to train a neural network

        Parameters
        ----------
        train_dataset : DataSet
            training dataset
        val_dataset : Union[None, DataSet], optional
            validation dataset, by default None
        save_path : Union[None, str], optional
            the path where the trained model should be saved, by default None
            #TODO: a callback for tensorboard and another for saving the model
        """
        super().train(train_dataset, val_dataset)
        self.params.update(kwargs)
        processed_x, processed_y = self.process_dataset(train_dataset, training=True)
        
        if val_dataset is not None:
            processed_x_val, processed_y_val = self.process_dataset(val_dataset, training=False)
            validation_data = (processed_x_val, processed_y_val)
        else:
            validation_data = None

        # init the model
        self.build_model()

        self._model.compile(optimizer=self._optimizer,
                            loss=self.params["loss"]["name"],
                            metrics=self.params["metrics"])        

        self.logger.info("Training of {%s} started", self.name)
        history_callback = self._model.fit(x=processed_x,
                                           y=processed_y,
                                           validation_data=validation_data,
                                           epochs=self.params["epochs"],
                                           batch_size=self.params["train_batch_size"],
                                           shuffle=self.params["shuffle"])
        self.logger.info("Training of {%s} finished", self.name)
        self.write_history(history=history_callback, val_dataset=validation_data)
        self.trained = True
        if save_path is not None:
            self.save(save_path)
        return history_callback

    def predict(self, dataset: DataSet, **kwargs) -> dict:
        """_summary_

        Parameters
        ----------
        dataset : DataSet
            test datasets to evaluate
        """
        super().predict(dataset)

        if "eval_batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["eval_batch_size"]
        # self.params.update(kwargs)

        #processed_x, processed_y = self._process_all_dataset(dataset, training=False)
        processed_x, _ = self.process_dataset(dataset, training=False)

        # make the predictions
        predictions = self._model.predict(processed_x, batch_size=self.params["eval_batch_size"])

        predictions = self._post_process(dataset, predictions)

        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = dataset.data

        return predictions

    def process_dataset(self, dataset: DataSet, training: bool) -> tuple:
        """process the datasets for training and evaluation

        each augmented simulator requires its owan data preparation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            _description_
        training : bool, optional
            _description_, by default False

        Returns
        -------
        tuple
            the normalized dataset with features and labels
        """
        super().process_dataset(dataset, training)
        inputs, outputs = dataset.extract_data()

        return inputs, outputs

    def _post_process(self, dataset, predictions):
        """Do some post processing on the predictions

        Parameters
        ----------
        predictions : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return dataset.reconstruct_output(predictions)


    ###############################################
    # function used to save and restore the model #
    ###############################################
    def save(self, path: str, save_metadata: bool=True):
        """_summary_

        Parameters
        ----------
        path : str
            _description_
        save_metadata : bool, optional
            _description_, by default True
        """
        save_path =  pathlib.Path(path) / self.name
        super().save(save_path)

        self._save_model(save_path)

        if save_metadata:
            self._save_metadata(save_path)

        self.logger.info("Model {%s} is saved at {%s}", self.name, save_path)

    def _save_model(self, path: Union[str, pathlib.Path], ext: str=".h5"):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        file_name = path / ("weights" + ext)
        self._model.save_weights(file_name)

    def _save_metadata(self, path: Union[str, pathlib.Path]):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        # for json serialization of paths
        #pydantic.json.ENCODERS_BY_TYPE[pathlib.PosixPath] = str
        #pydantic.json.ENCODERS_BY_TYPE[pathlib.WindowsPath] = str
        self._save_losses(path)
        with open((path / "config.json"), "w", encoding="utf-8") as f:
            json.dump(obj=self.params, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def restore(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        full_path = path / self.name
        if not full_path.exists():
            raise FileNotFoundError(f"path {full_path} not found")
        # load the metadata
        self._load_metadata(full_path)
        self._load_model(full_path)

        self.logger.info("Model {%s} is loaded from {%s}", self.name, full_path)

    def _load_model(self, path: str):
        nm_file = "weights.h5"
        path_weights = path / nm_file
        if not path_weights.exists():
            raise FileNotFoundError(f"Weights file {path_weights} not found")
        self.build_model()
        # load the weights
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(path_weights, nm_tmp)
            # load this copy (make sure the proper file is not corrupted even if the loading fails)
            self._model.load_weights(nm_tmp)

    def _load_metadata(self, path: str):
        """
        load the model metadata
        """
        # load scaler parameters
        #self.scaler.load(full_path)
        self._load_losses(path)
        with open((path / "config.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.params.update(res_json)
        return self.params

    def _save_losses(self, path: Union[str, pathlib.Path]):
        """
        save the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        res_losses = {}
        res_losses["train_losses"] = self.train_losses
        res_losses["train_metrics"] = self.train_metrics
        res_losses["val_losses"] = self.val_losses
        res_losses["val_metrics"] = self.val_metrics
        with open((path / "losses.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_losses, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_losses(self, path: Union[str, pathlib.Path]):
        """
        load the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "losses.json"), "r", encoding="utf-8") as f:
            res_losses = json.load(fp=f)
        self.train_losses = res_losses["train_losses"]
        self.train_metrics = res_losses["train_metrics"]
        self.val_losses = res_losses["val_losses"]
        self.val_metrics = res_losses["val_metrics"]

    #########################
    # Some Helper functions #
    #########################
    def summary(self):
        """summary of the model
        """
        print(self._model.summary())

    def plot_model(self, path: Union[str, None]=None, file_name: str="model"):
        """Plot the model architecture using GraphViz Library

        """
        # verify if GraphViz and pydot are installed
        pydot_found = importlib.util.find_spec("pydot")
        graphviz_found = importlib.util.find_spec("graphviz")
        if pydot_found is None or graphviz_found is None:
            raise RuntimeError("pydot and graphviz are required to use this function")

        if not pathlib.Path(path).exists():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        tf.keras.utils.plot_model(
            self._model,
            to_file=file_name+".png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=56,
            layer_range=None,
            show_layer_activations=False,
        )

    def write_history(self, history: dict, val_dataset=None):
        """write the history of the training

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        self.train_losses = history.history["loss"]
        if val_dataset is not None:
            self.val_losses = history.history["val_loss"]

        for metric in self.params["metrics"]:
            self.train_metrics[metric] = history.history[metric]
            if val_dataset is not None:
                self.val_metrics[metric] = history.history["val_" + metric]

    def count_parameters(self):
        """count the number of parameters of the model

        Returns
        -------
        int
            the number of parameters
        """
        return self._model.count_params()

    def visualize_convergence(self, figsize=(15,5), save_path: str=None):
        """Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = len(self.params["metrics"])
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)
        ax[0].set_title("MSE")
        ax[0].plot(self.train_losses, label='train_loss')
        if len(self.val_losses) > 0:
            ax[0].plot(self.val_losses, label='val_loss')
        for idx_, metric_name in enumerate(self.params["metrics"]):
            ax[idx_+1].set_title(metric_name)
            ax[idx_+1].plot(self.train_metrics[metric_name], label=f"train_{metric_name}")
            if len(self.val_metrics[metric_name]) > 0:
                ax[idx_+1].plot(self.val_metrics[metric_name], label=f"val_{metric_name}")
        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
