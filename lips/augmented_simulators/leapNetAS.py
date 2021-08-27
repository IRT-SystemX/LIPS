# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
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
from typing import Union, Dict
import tempfile
from tqdm import tqdm
import shutil

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.optimizers as tfko

from leap_net.proxy import ProxyLeapNet

from lips.dataset import DataSet
from lips.augmented_simulators.augmentedSimulator import AugmentedSimulator


class LeapNetAS(AugmentedSimulator):
    """
    This class wraps the `ProxyLeapNet` of the `leap_net` module (see https://github.com/BDonnot/leap_net) to be
    used as an `AugmentedSimulator`.

    This is just a wrapper, for modification of said "leap_net" or its description, please visit the original
    github repository.
    """
    def __init__(self,
                 name: str = "LeapNetAS",
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_tau=("line_status", "topo_vect"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 sizes_layer=(150, 150),
                 sizes_enc=(20, 20, 20),
                 sizes_out=(100, 40),
                 lr: float = 3e-4,  # only used if "optimizer" is not None
                 layer: Layer = Dense,
                 layer_act: str = "relu",
                 optimizer: Union[Optimizer, None] = None,
                 loss: str = "mse",  # loss used to train the model
                 batch_size: int = 128,
                 topo_vect_to_tau="raw",  # see code for now
                 #kwargs_tau: optional kwargs depending on the method chosen for building tau from the observation
                 kwargs_tau=None,
                 mult_by_zero_lines_pred=True
                ):

        AugmentedSimulator.__init__(self, name)
        self._attr_x = copy.deepcopy(attr_x)
        self._attr_tau = copy.deepcopy(attr_tau)
        self._attr_y = copy.deepcopy(attr_y)

        self.sizes_layer = copy.deepcopy(sizes_layer)
        self.sizes_enc = copy.deepcopy(sizes_enc)
        self.sizes_out = copy.deepcopy(sizes_out)

        # do i multiply by 0 the output of the powerlines to make sure that i got 0. as predictions when a
        # powerline is disconnected ? (if you do so make sure that the `_m_y` corresponding to lines
        # attributes are set to 0. too !
        self._mult_by_zero_lines_pred = mult_by_zero_lines_pred

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

        # ways to encode the topology vector
        self._topo_vect_to_tau = topo_vect_to_tau
        self._kwargs_tau = kwargs_tau

        # this is the proxy (from the leap net repo) "model"
        self._leap_net_model: Union[ProxyLeapNet, None] = None
        self._create_proxy()

    def init(self, **kwargs):
        """this function will build the neural network"""
        self._leap_net_model.build_model()

    def train(self, nb_iter: int, train_dataset: DataSet, val_dataset: Union[None, DataSet] = None):
        """This is an example of a reference implementation of this class. Feel"""
        # extract the input and output suitable for learning (matrices) from the generic dataset
        processed_x, processed_tau, processed_y = self._process_all_dataset(train_dataset, training=True)
        processed_x_val, processed_tau_val, processed_y_val = self._process_all_dataset(val_dataset, training=False)

        # create the neural network (now that I know the sizes)
        self.init()

        # 'compile' the keras model (now that it is initialized)
        # This is used to offer more customization compared to the original implementation
        self._leap_net_model._model.compile(optimizer=self._optimizer,
                                            loss=self._loss)

        # train the model
        self._leap_net_model._model.fit(x=(processed_x, processed_tau),
                                        y=processed_y,
                                        validation_data=((processed_x_val, processed_tau_val), processed_y_val),
                                        epochs=nb_iter,
                                        batch_size=self._batch_size)
        # NB in this function we use the high level keras method "fit" to fit the data. It does not stricly
        # uses the `DataSet` interface. For more complicated training loop, one can always use
        # dataset.get_data(indexes) to retrieve the batch of data corresponding to `indexes` and
        # `self.process_dataset` to process the example of this dataset one by one.

    def evaluate(self, dataset: DataSet):
        """evaluate the model on the given dataset"""
        # process the dataset
        processed_x, processed_tau, _ = self._process_all_dataset(dataset, training=False)

        # and out of speed, i directly used the loaded model to make the predictions and unscale them
        # make the predictions
        tmp = self._leap_net_model._model.predict((processed_x, processed_tau))
        res = {}
        proxy = self._leap_net_model
        for attr_nm, arr_, m_, sd_ in zip(self._attr_y, tmp, proxy._m_y, proxy._sd_y):
            res[attr_nm] = (arr_ * sd_) + m_
        return res

    def save(self, path_out: str):
        """
        This saves the weights of the neural network.
        """
        if not os.path.exists(path_out):
            raise RuntimeError(f"The path {path_out} does not exists.")
        if self._leap_net_model is None:
            raise RuntimeError(f"The model is not initialized i cannot save it")

        full_path_out = os.path.join(path_out, self.name)
        if not os.path.exists(full_path_out):
            os.mkdir(full_path_out)
            # TODO logger

        if self._leap_net_model is not None:
            # save the weights
            self._leap_net_model.save_data(full_path_out, ext=".h5")

    def restore(self, path: str):
        """
        Restores the model from a saved one.

        We first copy the weights file into a temporary directory, and then load from this one. This is avoid
        file corruption in case the model fails to load.
        """
        full_path = os.path.join(path, self.name)
        self._leap_net_model.load_data(full_path, ext=".h5")

    def save_metadata(self, path_out: str):
        """
        This is used to save the meta data of the augmented simulator.

        In this case it saves the sizes, the scalers etc.

        The only difficulty here is that i need to serialize, as json, numpy arrays
        """
        res_json = self._leap_net_model.get_metadata()
        res_json["batch_size"] = int(self._batch_size)
        res_json["lr"] = float(self._lr)
        res_json["layer_act"] = str(self.layer_act)
        res_json["_loss"] = str(self._loss)

        full_path_out = os.path.join(path_out, self.name)
        if not os.path.exists(full_path_out):
            os.mkdir(full_path_out)
            # TODO logger

        with open(os.path.join(full_path_out, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True)

    def load_metadata(self, path: str):
        """this is used to load the meta parameters from the model"""
        full_path = os.path.join(path, self.name)
        with open(os.path.join(full_path, f"metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)

        self._batch_size = int(res_json["batch_size"])
        self._lr = float(res_json["lr"])
        self.layer_act = str(res_json["layer_act"])
        self._loss = str(res_json["_loss"])
        self._leap_net_model.load_metadata(res_json)

    def _process_all_dataset(self, dataset: DataSet, training: bool = False):
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
            obss = self._make_fake_obs(all_data)
            self._leap_net_model.init(obss)

        proxy = self._leap_net_model
        res_x = []
        res_tau = []
        res_y = []
        for attr_nm, m_, sd_ in zip(self._attr_x, proxy._m_x, proxy._sd_x):
            res_x.append((all_data[attr_nm] - m_) / sd_)
        for attr_nm, m_, sd_ in zip(self._attr_tau, proxy._m_tau, proxy._sd_tau):
            res_tau.append((all_data[attr_nm] - m_) / sd_)
        for attr_nm, m_, sd_ in zip(self._attr_y, proxy._m_y, proxy._sd_y):
            res_y.append((all_data[attr_nm] - m_) / sd_)
        return res_x, res_tau, res_y

    def _make_fake_obs(self, all_data: Dict[str, np.ndarray]):
        """
        the underlying _leap_net_model requires some 'class' structure to work properly. This convert the
        numpy dataset into these structures.

        Definitely not the most efficient way to process a numpy array...
        """
        class FakeObs(object):
            pass

        if "topo_vect" in all_data:
            setattr(FakeObs, "dim_topo", all_data["topo_vect"].shape[1])

        nb_row = all_data[next(iter(all_data.keys()))].shape[0]
        obss = [FakeObs() for k in range(nb_row)]
        for attr_nm in all_data.keys():
            arr_ = all_data[attr_nm]
            for ind in range(nb_row):
                setattr(obss[ind], attr_nm, arr_[ind, :])

        return obss

    def _create_proxy(self):
        """part of the "wrapper" part, this function will initialize the leap net model"""
        self._leap_net_model = ProxyLeapNet(
            name=f"{self.name}_model",
            train_batch_size=self._batch_size,
            attr_x=self._attr_x,
            attr_y=self._attr_y,
            attr_tau=self._attr_tau,
            sizes_enc=self.sizes_enc,
            sizes_main=self.sizes_layer,
            sizes_out=self.sizes_out,
            lr=self._lr,
            layer=self.layer,  # TODO (for save and restore)
            topo_vect_to_tau=self._topo_vect_to_tau,  # see code for now
            # kwargs_tau: optional kwargs depending on the method chosen for building tau from the observation
            kwargs_tau=self._kwargs_tau,
            mult_by_zero_lines_pred=self._mult_by_zero_lines_pred,
            # TODO there for AS
            scale_main_layer=None,  # increase the size of the main layer
            scale_input_dec_layer=None,  # scale the input of the decoder
            scale_input_enc_layer=None,  # scale the input of the encoder
            layer_act=None,
        )
