# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras

from leap_net import ResNetLayer

from .utils import TopoVectTransformation
from ..fully_connected import TfFullyConnected
from ....dataset import DataSet

class TfResNetPowerGrid(TfFullyConnected):
    """ResNet architecture

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define layer to be used for the model
        self.layers = {"resnet": ResNetLayer}
        self.layer = self.layers[self.params["layer"]]
        self._topo_vect_transformer = None

    def build_model(self):
        """Build the model
        Returns
        -------
        Model
            _description_
        """
        super(TfFullyConnected, self).build_model()

        input_ = keras.layers.Input(shape=(self.input_size,), name="input")
        x = input_
        x = keras.layers.Dropout(rate=self.params["input_dropout"], name="input_dropout")(x)

        if "scale_input_layer" in self.params and self.params["scale_input_layer"]:
            x = keras.layers.Dense(self.params["layers"][0], name="scaling_input_ResNet")(x)

        for layer_id, layer_size in enumerate(self.params["layers"]):
            x = self.layer(layer_size, name=f"layer_{layer_id}")(x)
            x = keras.layers.Activation(self.params["activation"], name=f"activation_{layer_id}")(x)
            x = keras.layers.Dropout(rate=self.params["dropout"], name=f"dropout_{layer_id}")(x)
        output_ = keras.layers.Dense(self.output_size)(x)
        self._model = keras.Model(inputs=input_,
                                  outputs=output_,
                                  name=f"{self.name}_model")
        return self._model
    
    def init_topo_vect_transformer(self, dataset: DataSet):
        """Initialize the topo_vect_transformer using the training dataset

        Args:
            dataset (DataSet): _description_
        """
        self._topo_vect_transformer = TopoVectTransformation(self.bench_config, self.params, dataset)

    def process_dataset(self, dataset: DataSet, training: bool = False) -> tuple:
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

        # extract inputs, line_status, outputs without concatenation
        (inputs, extract_tau), outputs = dataset.extract_data(concat=False)
        line_status = extract_tau[0]

        if training:
            self.init_topo_vect_transformer(dataset)
            #self._topo_vect_transformer = TopoVectTransformation(self.bench_config, self.params, dataset)
        else:
            if self._topo_vect_transformer is None:
                raise RuntimeError(f"If you load a pretrained model, you should first call the `init_topo_vect_transformer` function on training data!")

        # extract tau using LeapNetProxy function
        extract_tau = self._topo_vect_transformer.transform_topo_vect(dataset)

        # add tau and line_status to inputs
        inputs.extend([extract_tau, line_status])

        # concatenate input features
        inputs = np.concatenate([el.astype(np.float32) for el in inputs], axis=1)

        # concatenate outputs labels
        outputs = np.concatenate([el.astype(np.float32) for el in outputs], axis=1)

        if training:
            # set input and output sizes
            self.input_size = inputs.shape[1]
            self.output_size = outputs.shape[1]
            #TODO : exclude scaling line_status and tau features
            if self.scaler is not None:
                inputs, outputs = self.scaler.fit_transform(inputs, outputs)
        else:
            if self.scaler is not None:
                inputs, outputs = self.scaler.transform(inputs, outputs)
        return inputs, outputs
