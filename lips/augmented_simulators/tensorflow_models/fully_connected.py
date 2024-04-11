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
import json
import warnings

import numpy as np
# from leap_net import ResNetLayer

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras

from ..tensorflow_simulator import TensorflowSimulator
from ...logger import CustomLogger
from ...config import ConfigManager
from ...dataset import DataSet
from ...dataset.scaler import Scaler
from ...utils import NpEncoder

class TfFullyConnected(TensorflowSimulator):
    """Fully Connected architecture
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
                 bench_config_name: Union[str, None]=None,
                 bench_kwargs: dict={},
                 sim_config_name: Union[str, None]=None,
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
        self.name = self.name + '_' + sim_config_name
        # scaler
        self.scaler = scaler() if scaler else None
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)
        # Define layer to be used for the model
        self.layers = {"linear": keras.layers.Dense}#, "resnet" : ResNetLayer}
        self.layer = self.layers.get(self.params["layer"], None)
        if self.layer is None:
            self.layer = keras.layers.Dense 

        # optimizer
        if "optimizer" in kwargs:
            if not isinstance(kwargs["optimizer"], keras.optimizers.Optimizer):
                raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = kwargs["optimizer"](self.params["optimizer"]["params"])
        else:
            self._optimizer = keras.optimizers.Adam(learning_rate=self.params["optimizer"]["params"]["lr"])

        self._model: Union[keras.Model, None] = None

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]

    def build_model(self):
        """Build the model
        Returns
        -------
        Model
            _description_
        """
        super().build_model()
        input_ = keras.layers.Input(shape=(self.input_size,), name="input")
        x = input_
        x = keras.layers.Dropout(rate=self.params["input_dropout"], name="input_dropout")(x)
        for layer_id, layer_size in enumerate(self.params["layers"]):
            x = self.layer(layer_size, name=f"layer_{layer_id}")(x)
            x = keras.layers.Activation(self.params["activation"], name=f"activation_{layer_id}")(x)
            x = keras.layers.Dropout(rate=self.params["dropout"], name=f"dropout_{layer_id}")(x)
        output_ = keras.layers.Dense(self.output_size)(x)
        self._model = keras.Model(inputs=input_,
                                  outputs=output_,
                                  name=f"{self.name}_model")
        return self._model

    def process_dataset(self, dataset: DataSet, training: bool=False) -> tuple:
        """process the datasets for training and evaluation
        This function transforms all the dataset into something that can be used by the neural network (for example)
        Warning
        -------
        It works with StandardScaler only for the moment.
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
            self._infer_size(dataset)
            inputs, outputs = dataset.extract_data(concat=True)
            if self.scaler is not None:
                inputs, outputs = self.scaler.fit_transform(inputs, outputs)
        else:
            inputs, outputs = dataset.extract_data(concat=True)
            if self.scaler is not None:
                inputs, outputs = self.scaler.transform(inputs, outputs)

        return inputs, outputs

    def _infer_size(self, dataset: DataSet):
        """Infer the size of the model
        Parameters
        ----------
        dataset : DataSet
            _description_
        Returns
        -------
        None
            _description_
        """
        *dim_inputs, self.output_size = dataset.get_sizes()
        self.input_size = np.sum(dim_inputs)

    def _post_process(self, dataset, predictions):
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        predictions = super()._post_process(dataset, predictions)
        return predictions

    def _save_metadata(self, path: str):
        super()._save_metadata(path)
        if self.scaler is not None:
            self.scaler.save(path)
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        super()._load_metadata(path)
        if self.scaler is not None:
            self.scaler.load(path)
        with open((path / "metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.input_size = res_json["input_size"]
        self.output_size = res_json["output_size"]
