"""
Tensorflow based augmented simulators
"""
import os
import pathlib
from typing import Union
import shutil
import time
import json
import tempfile
import warnings
import importlib

import numpy as np
from matplotlib import pyplot as plt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

from . import AugmentedSimulator
from .tensorflow_models.utils import NpEncoder
from ..dataset import DataSet
from ..dataset import Scaler
from ..logger import CustomLogger

class TensorflowSimulator(AugmentedSimulator):
    """_summary_

        Parameters
        ----------
        model : Union[Model, Sequential]
            _description_
        name : str, optional
            _description_, by default None
        scaler : Scaler, optional
            scaler used to scale the data, by default None
        config : ConfigManager
            _description_
        """
    def __init__(self,
                 model: keras.Model,
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 log_path: Union[str, None] = None,
                 **kwargs):
        super().__init__(model, name, scaler, log_path, **kwargs)
        # logger
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        self._model = self._build_model(**kwargs)

        if name is not None:
            self.name = name
        else:
            self.name = self._model.name

        # optimizer
        if "optimizer" in kwargs:
            if not isinstance(kwargs["optimizer"], keras.optimizers.Optimizer):
                raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = kwargs["optimizer"](self.params["optimizers"]["params"])
        else:
            self._optimizer = keras.optimizers.Adam(learning_rate=self.params["optimizers"]["params"]["lr"])

    def _build_model(self, **kwargs) -> keras.Model:
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
        model_tmp = self.model(**kwargs)
        self.params.update(model_tmp.params)
        return model_tmp._model

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
        self.params.update(kwargs)
        processed_x, processed_y = self._process_all_dataset(train_dataset, training=True)
        if val_dataset is not None:
            processed_x_val, processed_y_val = self._process_all_dataset(val_dataset, training=False)

        self._model.compile(optimizer=self._optimizer,
                            loss=self.params["loss"]["name"],
                            metrics=self.params["metrics"])

        if val_dataset is not None:
            validation_data = (processed_x_val, processed_y_val)
        else:
            validation_data = None

        self.logger.info("Training of {%s} started", self.name)
        history_callback = self._model.fit(x=processed_x,
                                           y=processed_y,
                                           validation_data=validation_data,
                                           epochs=self.params["epochs"],
                                           batch_size=self.params["train_batch_size"],
                                           shuffle=self.params["shuffle"])
        self.logger.info("Training of {%s} finished", self.name)
        self.write_history(history_callback)
        self.trained = True
        if save_path is not None:
            self.save(save_path)
        return history_callback

    def evaluate(self, dataset: DataSet, **kwargs) -> dict:
        """_summary_

        Parameters
        ----------
        dataset : DataSet
            test datasets to evaluate
        """
        if "batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["batch_size"]
        self.params.update(kwargs)

        processed_x, processed_y = self._process_all_dataset(dataset, training=False)

        # make the predictions
        _beg = time.time()
        tmp_res_y = self._model.predict(processed_x, batch_size=self.params["eval_batch_size"])
        self.predict_time = time.time() - _beg

        if self.scaler is not None:
            observations = self.scaler.inverse_transform(processed_y)
            predictions = self.scaler.inverse_transform(tmp_res_y)

        predictions = dataset.reconstruct_output(predictions)
        observations = dataset.reconstruct_output(observations)

        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = dataset.data

        return predictions


    def _process_all_dataset(self, dataset: DataSet, training: bool=False) -> tuple:
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            _description_
        Scaler : bool, optional
            _description_, by default True
        training : bool, optional
            _description_, by default False

        Returns
        -------
        tuple
            the normalized dataset with features and labels
        """
        if training:
            extract_x, extract_y = dataset.extract_data()
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
        else:
            extract_x, extract_y = dataset.extract_data()
            if dataset._size_x is None:
                raise RuntimeError("Model cannot be used, we don't know the size of the input vector. Either train it "
                                "or load its meta data properly.")
            if dataset._size_y is None:
                raise RuntimeError("Model cannot be used, we don't know the size of the output vector. Either train it "
                                "or load its meta data properly.")
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

        return extract_x, extract_y

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

        file_name = save_path / ("model" + ".h5")
        self._model.save(file_name)

        if save_metadata:
            self._save_metadata(save_path)

        self.logger.info("Model {%s} is saved at {%s}", self.name, save_path)

    def _save_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self.scaler.save(path)
        self._save_losses(path)
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=self.params, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def restore(self, path: str):

        nm_file = "model.h5"
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        path_weights = path / self.name / nm_file
        if not path_weights.exists():
            raise FileNotFoundError(f"Weights file {path_weights} not found")
        # load the metadata
        params = self._load_metadata(path)
        # build the model
        self._model = self._build_model(**params)
        # load the weights
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(path_weights, nm_tmp)
            # load this copy (make sure the proper file is not corrupted even if the loading fails)
            self._model.load_weights(nm_tmp)

        self.logger.info("Model {%s} is loaded from {%s}", self.name, path_weights)

    def _load_metadata(self, path: str):
        """
        load the model metadata
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        full_path = path / self.name
        # load scaler parameters
        self.scaler.load(full_path)
        self._load_losses(full_path)
        with open((full_path / "metadata.json"), "r", encoding="utf-8") as f:
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

    def write_history(self, history_callback):
        """write the history of the training

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        self.train_losses = history_callback.history["loss"]
        self.val_losses = history_callback.history["val_loss"]

        for metric in self.params["metrics"]:
            self.train_metrics[metric] = history_callback.history[metric]
            self.val_metrics[metric] = history_callback.history["val_" + metric]

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