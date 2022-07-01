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

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras

try:
    from leap_net.proxy import ProxyLeapNet
except ImportError as err:
    raise RuntimeError("You need to install the leap_net package to use this class") from err


from ..tensorflow_simulator import TensorflowSimulator
from ...logger import CustomLogger
from ...config import ConfigManager
from ...dataset import DataSet
from ...dataset.scaler import Scaler


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
                 sim_config_name: Union[str, None]=None,
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 bench_config_path: Union[str, pathlib.Path, None]=None,
                 bench_config_name: Union[str, None]=None,
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
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)
        # optimizer
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
        self._leap_net_model = ProxyLeapNet(
            name=f"{self.name}_model",
            train_batch_size=self.params["train_batch_size"],
            attr_x=self.bench_config.get_option("attr_x"),
            attr_y=self.bench_config.get_option("attr_y"),
            attr_tau=self.bench_config.get_option("attr_tau"),
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
        if training:
            obss = self._make_fake_obs(dataset)
            self._leap_net_model.init(obss)
            if self.scaler is not None:
                (extract_x, extract_tau), extract_y = self.scaler.fit_transform(dataset)
            else:
                (extract_x, extract_tau), extract_y = dataset.extract_data(concat=False)
            extract_tau = self._transform_tau(dataset, extract_tau)
        else:
            if self.scaler is not None:
                (extract_x, extract_tau), extract_y = self.scaler.transform(dataset)
            else:
                (extract_x, extract_tau), extract_y = dataset.extract_data(concat=False)
            extract_tau = self._transform_tau(dataset, extract_tau)

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
        obss = self._make_fake_obs(dataset)
        tau.pop()
        tau.append(np.array([self._leap_net_model.topo_vect_handler(obs)
                            for obs in obss],
                            dtype=np.float32))
        return tau

    def _make_fake_obs(self, dataset: DataSet):
        """
        the underlying _leap_net_model requires some 'class' structure to work properly. This convert the
        numpy dataset into these structures.

        Definitely not the most efficient way to process a numpy array...
        """
        all_data = dataset.data
        class FakeObs(object):
            pass

        if "topo_vect" in all_data:
            setattr(FakeObs, "dim_topo", all_data["topo_vect"].shape[1])

        setattr(FakeObs, "n_sub", dataset.env_data["n_sub"])
        setattr(FakeObs, "sub_info", np.array(dataset.env_data["sub_info"]))

        nb_row = all_data[next(iter(all_data.keys()))].shape[0]
        obss = [FakeObs() for k in range(nb_row)]
        for attr_nm in all_data.keys():
            arr_ = all_data[attr_nm]
            for ind in range(nb_row):
                setattr(obss[ind], attr_nm, arr_[ind, :])
        return obss

    def write_history(self, history):
        """write the history of the training

        It should have its own history writer as it includes multiple variables

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        hist = history.history

        self.train_losses = hist["loss"]
        self.val_losses = hist["val_loss"]

        for metric in ["mae"]:
            tmp_train = []
            tmp_val = []
            for key in hist.keys():
                if (metric in key) and ("val" in key):
                    tmp_val.append(hist[key])
                if (metric in key) and ("val" not in key):
                    tmp_train.append(hist[key])
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


