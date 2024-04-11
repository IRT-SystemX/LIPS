# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""
The leap_net model
"""
import os
import pathlib
from typing import Union
import json
import warnings
import copy

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras
    import tensorflow as tf

try:
    from leap_net.proxy import ProxyLeapNet
except ImportError as err:
    raise RuntimeError("You need to install the leap_net package to use this class") from err

from .utils import TopoVectTransformation
from ...tensorflow_simulator import TensorflowSimulator
from ....logger import CustomLogger
from ....config import ConfigManager
from ....dataset import DataSet
from ....dataset.scaler import Scaler


class LeapNet(TensorflowSimulator):
    """LeapNet architecture

    This class wraps the `ProxyLeapNet` of the `leap_net` module (see https://github.com/BDonnot/leap_net) to be
    used as an `AugmentedSimulator`.

    This is just a wrapper, for modification of said "leap_net" or its description, please visit the original
    github repository.


    Parameters
    ----------
    sim_config_path : ``str``
        The path to the configuration file for simulator.
        It should contain all the required hyperparameters for this model.
    sim_config_name : Union[str, None], optional
        _description_, by default None
    name : Union[str, None], optional
        _description_, by default None
    scaler : Union[Scaler, None], optional
        _description_, by default None
    bench_config_path : Union[str, pathlib.Path, None], optional
        _description_, by default None
    bench_config_name : Union[str, None], optional
        _description_, by default None
    log_path : Union[None, str], optional
        _description_, by default None

    Raises
    ------
    RuntimeError
        _description_
    """
    def __init__(self,
                 sim_config_path: str,
                 bench_config_path: Union[str, pathlib.Path],
                 sim_config_name: Union[str, None]=None,
                 bench_config_name: Union[str, None]=None,
                 bench_kwargs: dict = {},
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 log_path: Union[None, str]=None,
                 **kwargs):

        super().__init__(name=name, log_path=log_path, **kwargs)
        if not os.path.exists(sim_config_path):
            raise RuntimeError("Configuration path for the simulator not found!")
        if not str(sim_config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        self.bench_config.set_options_from_dict(**bench_kwargs)
        self.name = name if name is not None else self.sim_config.get_option("name")
        self.name = name + '_' + sim_config_name
        # scaler
        self.scaler = scaler() if scaler else None
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # Define layer to be used for the model
        self.layers = {"linear": keras.layers.Dense}
        self.layer = self.layers[self.sim_config.get_option("layer")]
        # get tau attributes
        self.attr_tau = kwargs['attr_tau'] if "attr_tau" in kwargs else self.bench_config.get_option("attr_tau")
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)

        # optimizer
        if "optimizer" in kwargs:
            if not isinstance(kwargs["optimizer"], keras.optimizers.Optimizer):
                raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = kwargs["optimizer"](self.params["optimizer"]["params"])
        else:
            self._optimizer = keras.optimizers.Adam(learning_rate=self.params["optimizer"]["params"]["lr"])
        # Init the leap net model
        self._leap_net_model: Union[ProxyLeapNet, None] = None
        self._model: Union[keras.Model, None] = None
        self._create_proxy()

    def build_model(self):
        """Build the model"""
        self._leap_net_model.build_model()
        self._model = self._leap_net_model._model

    def _create_proxy(self):
        """part of the "wrapper" part, this function will initialize the leap net model"""
        # TODO add comment
        attr_tau = ("concatenated_tau",) if "concatenate_tau" in self.params and self.params["concatenate_tau"]\
                       else self.bench_config.get_option("attr_tau")

        self._leap_net_model = ProxyLeapNet(
            name=f"{self.name}_model",
            train_batch_size=self.params["train_batch_size"],
            attr_x=self.bench_config.get_option("attr_x"),
            attr_y=self.bench_config.get_option("attr_y"),
            attr_tau=attr_tau,
            sizes_enc=self.params["sizes_enc"],
            sizes_main=self.params["sizes_main"],
            sizes_out=self.params["sizes_out"],
            lr=self.params["optimizer"]["params"]["lr"],
            layer=self.layer,  # TODO (for save and restore)
            topo_vect_to_tau=self.params["topo_vect_to_tau"],  # see code for now
            # kwargs_tau: optional kwargs depending on the method chosen for building tau from the observation
            kwargs_tau=self.params["kwargs_tau"],
            mult_by_zero_lines_pred=self.params["mult_by_zero_lines_pred"],
            # TODO there for AS
            scale_main_layer=self.params["scale_main_layer"],  # increase the size of the main layer
            scale_input_dec_layer=self.params["scale_input_dec_layer"],  # scale the input of the decoder
            scale_input_enc_layer=self.params["scale_input_enc_layer"],  # scale the input of the encoder
            layer_act=self.params["activation"]
        )

    def process_dataset(self, dataset: DataSet, training: bool=False) -> tuple:
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Warning
        -------
        It works only with PowerGridScaler scaler for the moment

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
        # deep copy to avoid changing the original dataset argument
        dataset_copy = copy.deepcopy(dataset)

        # extract and transform topo_vect and inject it into the dataset
        if "topo_vect" in dataset.data:
            if training :
                self._topo_vect_transformer = TopoVectTransformation(self.bench_config, self.params, dataset)
            transformed_topo_vect = self._topo_vect_transformer.transform_topo_vect(dataset)

        # concatenate line_status and topo_vect into a single feature, if the concatenate_tau param is enabled
        if "concatenate_tau" in self.params and self.params["concatenate_tau"]:
            if all(el in self.bench_config.get_option("attr_tau") for el in ("line_status", "topo_vect")):
                dataset_copy.data["concatenated_tau"] = \
                    np.concatenate((dataset.data["line_status"], transformed_topo_vect), axis=1)
            else: raise RuntimeError("line_status or topo_vect not found in attr_tau argument, "
                                         "please add them in benchmark config file")

        if training:
            obss = self._topo_vect_transformer._make_fake_obs(dataset_copy)
            self._leap_net_model.init(obss)

            # replace the old topo_vect by the transformed one
            dataset_copy.data["topo_vect"] = transformed_topo_vect

            if self.scaler is not None:
                (extract_x, extract_tau), extract_y = self.scaler.fit_transform(dataset_copy)
            else:
                (extract_x, extract_tau), extract_y = dataset_copy.extract_data(concat=False)

        else:
            dataset_copy.data["topo_vect"] = transformed_topo_vect

            if self.scaler is not None:
                (extract_x, extract_tau), extract_y = self.scaler.transform(dataset_copy)
            else:
                (extract_x, extract_tau), extract_y = dataset_copy.extract_data(concat=False)

        if "concatenate_tau" in self.params and self.params["concatenate_tau"]:
            extract_tau = np.concatenate(extract_tau, axis=1)

        return (extract_x, extract_tau), extract_y

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
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def _transform_tau(self, dataset, tau):
        """Transform only the tau vector with respect to LeapNet encodings
        """
        obss = self._topo_vect_transformer._make_fake_obs(dataset)
        tau.pop()
        tau.append(np.array([self._leap_net_model.topo_vect_handler(obs)
                            for obs in obss],
                            dtype=np.float32))
        return tau

    def write_history(self, history, val_dataset=None):
        """write the history of the training

        It should have its own history writer as it includes multiple variables

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        hist = history.history

        self.train_losses = hist["loss"]
        if val_dataset is not None:
            self.val_losses = hist["val_loss"]
        else:
            self.val_losses = None
        metrics=self.params["metrics"]

        for metric in metrics:#["mae"]:
            tmp_train = []
            tmp_val = []
            for key in hist.keys():
                if (metric in key) and ("val" in key):
                    tmp_val.append(hist[key])
                if (metric in key) and ("val" not in key):
                    tmp_train.append(hist[key])
            if val_dataset is not None:
                self.val_metrics[metric] = np.vstack(tmp_val).mean(axis=0)
            self.train_metrics[metric] = np.vstack(tmp_train).mean(axis=0)

    def _save_model(self, path: str):
        if self._leap_net_model is not None:
            # save the weights
            self._leap_net_model.save_data(path, ext=".h5")

    def _load_model(self, path: str):
        self._leap_net_model.build_model()
        self._leap_net_model.load_data(path, ext=".h5")
        self._model = self._leap_net_model._model

    def _save_metadata(self, path: str):
        super()._save_metadata(path)

        res_json = self._leap_net_model.get_metadata()
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True)

        if self.scaler is not None:
            self.scaler.save(path)

    def _load_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        super()._load_metadata(path)
        with open(os.path.join(path, f"metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self._leap_net_model.load_metadata(res_json)

        if self.scaler is not None:
            self.scaler.load(path)


