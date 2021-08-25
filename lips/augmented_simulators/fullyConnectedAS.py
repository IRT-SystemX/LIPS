# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import time
import json
import numpy as np
import copy
import warnings
from typing import Union
import tempfile
from tqdm import tqdm
import shutil

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.optimizers as tfko

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Input

from lips.augmented_simulators import AugmentedSimulator


class FullyConnectedAS(AugmentedSimulator):
    """
    This class uses tensorflow and keras to implement an Augmented simulator based on a fully connected neural network
    trained using keras.

    What it does is simply to concatenate all the input attributes together in one matrix, is asked to output
    all the output in one big matrix.

    It also handles basic scaling of the data

    "FullyConnectedAS" stands for "Fully Connected neural network used as an Agumented Simulator"
    """
    def __init__(self, 
                 name: str = "FullyConnected",
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 sizes_layer=(150, 150),
                 lr: float = 3e-4,  # only used if "optimizer" is not None
                 layer: Layer = Dense,
                 layer_act: str = "relu",
                 optimizer: Union[Optimizer, None] = None,
                 loss: str = "mse",  # loss used to train the model
                 batch_size: int = 128,
                ):

        AugmentedSimulator.__init__(self, name)
        self._attr_x = copy.deepcopy(attr_x)
        self._attr_y = copy.deepcopy(attr_y)
        self.sizes_layer = copy.deepcopy(sizes_layer)
        self._lr = lr
        self.layer = layer
        self.layer_act = layer_act
        self._loss = loss
        self._batch_size = batch_size
        if optimizer is not None:
            if not isinstance(optimizer, Optimizer):
                raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = optimizer
        else:
            self._optimizer = tfko.Adam(learning_rate=self._lr)

        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_y = None
        self._sizes_x = None  # dimension of each variable
        self._sizes_y = None  # dimension of each variable
        # this model normalizes the data by dividing by the variance and removing the means, i need to keep them
        self._std_x = None
        self._std_y = None
        self._m_x = None
        self._m_y = None
        # this is the keras "model"
        self._model: Union[Model, None] = None

    def init(self, **kwargs):
        """this function will build the neural network"""
        if self._model is not None:
            # model is already initialized
            return
        if self._size_x is None or self._size_y is None:
            raise RuntimeError("Impossible to initialize the model if i don't know its size. Have you called "
                               "`fully_connected.load_metada` or `fully_connected.train` ?")
        self._model = Sequential()
        input_ = Input(shape=(self._size_x,), name="input")

        # now make the model
        previous = input_
        for layer_id, layer_size in enumerate(self.sizes_layer):
            previous = self.layer(layer_size, name=f"layer_{layer_id}")(previous)
            previous = Activation(self.layer_act, name=f"activation_{layer_id}")(previous)
        output_ = Dense(self._size_y)(previous)
        self._model = Model(inputs=input_,
                            outputs=output_,
                            name=f"{self.name}_model")

    def train(self, dataset, nb_iter):
        """This is an example of a reference implementation of this class. Feel"""
        # extract the input and output suitable for learning (matrices) from the generic dataset
        processed_x, processed_y = self._process_all_dataset(dataset, training=True)

        # create the neural network (now that I know the sizes)
        self.init()

        # 'compile' the keras model (now that it is initialized)
        self._model.compile(optimizer=self._optimizer,
                            loss=self._loss)

        # train the model
        self._model.fit(x=processed_x,
                        y=processed_y,
                        epochs=nb_iter,
                        batch_size=self._batch_size)
        # NB in this function we use the high level keras method "fit" to fit the data. It does not stricly
        # uses the `DataSet` interface. For more complicated training loop, one can always use
        # dataset.get_data(indexes) to retrieve the batch of data corresponding to `indexes` and
        # `self.process_dataset` to process the example of this dataset one by one.

    def evaluate(self, dataset):
        """evaluate the model on the given dataset"""
        # process the dataset
        processed_x, _ = self._process_all_dataset(dataset, training=False)

        # make the predictions
        tmp_res_y = self._model.predict(processed_x)
        # rescale them
        tmp_res_y *= self._std_y
        tmp_res_y += self._m_y

        # and now output data as a dictionary
        res = {}
        prev_ = 0
        for var_id, this_var_size in enumerate(self._sizes_y):
            attr_nm = self._attr_y[var_id]
            res[attr_nm] = tmp_res_y[:, prev_:(prev_ + this_var_size)]
            prev_ += this_var_size
        return res

    def save(self, path_out):
        """
        This saves the weights of the neural network.
        """
        if not os.path.exists(path_out):
            raise RuntimeError(f"The path {path_out} does not exists.")
        if self._model is None:
            raise RuntimeError(f"The model is not initialized i cannot save it")

        full_path_out = os.path.join(path_out, self.name)
        if not os.path.exists(full_path_out):
            os.mkdir(full_path_out)
            # TODO logger

        if self._model is not None:
            # save the weights
            self._model.save(os.path.join(full_path_out, "model.h5"))

    def restore(self, path):
        """
        Restores the model from a saved one.

        We first copy the weights file into a temporary directory, and then load from this one. This is avoid
        file corruption in case the model fails to load.
        """
        nm_file = f"model.h5"
        path_weights = os.path.join(path, self.name, nm_file)
        if not os.path.exists(path_weights):
            raise RuntimeError(f"Impossible to find a saved model named {self.name} at {path}")

        with tempfile.TemporaryDirectory() as path_tmp:
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(path_weights, nm_tmp)
            # load this copy (make sure the proper file is not corrupted even if the loading fails)
            self._model.load_weights(nm_tmp)

    def save_metadata(self, path_out):
        """
        This is used to save the meta data of the augmented simulator.

        In this case it saves the sizes, the scalers etc.

        The only difficulty here is that i need to serialize, as json, numpy arrays
        """
        res_json = {"batch_size": int(self._batch_size),
                    "lr": float(self._lr),
                    "layer_act": str(self.layer_act),
                    "_loss": str(self._loss),
                    "_size_x": int(self._size_x),
                    "_size_y": int(self._size_y)}
        for my_attr in ["_sizes_x", "_sizes_y", "_m_x", "_m_y", "_std_x",
                        "_std_y", "sizes_layer", "_attr_x", "_attr_y"]:
            tmp = getattr(self, my_attr)
            fun = lambda x: x
            if isinstance(tmp, np.ndarray):
                if tmp.dtype == int or tmp.dtype == np.int or tmp.dtype == np.int32 or tmp.dtype == np.int64:
                    fun = int
                elif tmp.dtype == float or tmp.dtype == np.float32 or tmp.dtype == np.float64:
                    fun = float
            res_json[my_attr] = [fun(el) for el in tmp]

        full_path_out = os.path.join(path_out, self.name)
        if not os.path.exists(full_path_out):
            os.mkdir(full_path_out)
            # TODO logger

        with open(os.path.join(full_path_out, f"{self.name}_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True)

    def load_metadata(self, path):
        """this is not used for now"""
        with open(os.path.join(path, f"{self.name}_metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)

        self._batch_size = int(res_json["batch_size"])
        self._lr = float(res_json["lr"])
        self.layer_act = str(res_json["layer_act"])
        self._loss = str(res_json["_loss"])
        self._attr_x = res_json["_attr_x"]
        self._attr_y = res_json["_attr_y"]
        self.sizes_layer = res_json["sizes_layer"]
        self._size_x = int(res_json["_size_x"])
        self._size_y = int(res_json["_size_y"])
        self._sizes_x = np.array(res_json["_sizes_x"], dtype=int)
        self._sizes_y = np.array(res_json["_sizes_y"], dtype=int)

        self._m_x = np.array(res_json["_m_x"], dtype=np.float32)
        self._m_y = np.array(res_json["_m_y"], dtype=np.float32)
        self._std_x = np.array(res_json["_std_x"], dtype=np.float32)
        self._std_y = np.array(res_json["_std_y"], dtype=np.float32)

    def _process_all_dataset(self, dataset, training=False):
        """This function will extract the whole dataset and format it in a way we can train the
        fully connected neural network from it

        if "training" is `True` then it will also computes the scalers:

        - _std_x
        - _std_y
        - _m_x
        - _m_y

        And the size of the dataset self._size_x and self._size_y
        """
        all_data = dataset.get_data(np.arange(len(dataset)))
        if training:
            # init the sizes and everything
            self._sizes_x = np.array([all_data[el].shape[1] for el in self._attr_x], dtype=int)
            self._sizes_y = np.array([all_data[el].shape[1] for el in self._attr_y], dtype=int)
            self._size_x = np.sum(self._sizes_x)
            self._size_y = np.sum(self._sizes_y)

        if self._size_x is None:
            raise RuntimeError("Model cannot be used, we don't know the size of the input vector. Either train it "
                               "or load its meta data properly.")
        if self._size_y is None:
            raise RuntimeError("Model cannot be used, we don't know the size of the output vector. Either train it "
                               "or load its meta data properly.")

        res_x = np.concatenate([all_data[el].astype(np.float32) for el in self._attr_x], axis=1)
        res_y = np.concatenate([all_data[el].astype(np.float32) for el in self._attr_y], axis=1)

        if training:
            self._m_x = np.mean(res_x, axis=0)
            self._m_y = np.mean(res_y, axis=0)
            self._std_x = np.std(res_x, axis=0)
            self._std_y = np.std(res_y, axis=0)

            # to avoid division by 0.
            self._std_x[np.abs(self._std_x) <= 1e-1] = 1
            self._std_y[np.abs(self._std_y) <= 1e-1] = 1

        if self._m_x is None or self._m_y is None or self._std_x is None or self._std_y is None:
            raise RuntimeError("Model cannot be used, we don't know the size of the output vector. Either train it "
                               "or load its meta data properly.")

        res_x -= self._m_x
        res_x /= self._std_x
        res_y -= self._m_y
        res_y /= self._std_y

        return res_x, res_y
