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
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import multiply as tfk_multiply
from tensorflow.keras import optimizers



with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Input

from lips.simulators import AugmentedSimulator

class FullyConnectedNN(AugmentedSimulator):
    """
    Full connected neural network
    """
    def __init__(self, 
                 name="FullyConnected", 
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "line_status"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 sizes_enc=[20, 20],
                 sizes_main=[150, 150],
                 sizes_out=[40, 40],
                 lr=3e-4,
                 layer=Dense,
                 layer_act=None
                ):

        AugmentedSimulator.__init__(self, name=name)

        self._line_attr = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "v_or", "v_ex"}
        self.attr_x = attr_x
        self.attr_y = attr_y
        
        self._x_preprocessed = None
        self._y_preprocessed = None
        
        self._model = None
        self.sizes_enc = sizes_enc
        self.sizes_main = sizes_main
        self.sizes_out = sizes_out
        self.lr = lr
        self.layer = layer
        self.layer_act = layer_act

        self._y_hat = []

        self.training_data = None
        self.test_data = None
        

        self._time_train = 0
        self.predict_time = 0
        self.loss_metric_dict_train = None
        self.loss_metric_dict_valid = None
        self.global_loss = list()

        self.loss_dicts_init=False

        self._idx = None
        self._where_id = None
        self.tensor_line_status = None

        try:
            self._idx = self.attr_x.index("line_status")
            self._where_id = "x"
        except ValueError:
            warnings.warn("We strongly recommend you to get the \"line_status\" as an input vector")


    def init(self, obs_list):
        super().init(obs_list)
        
    def init_loss_dicts(self):
        """
        initialize the losses dictionary from the built and trained model

        a loss value is associated to each output variable
        """
        # initialize loss dictionaries with losses from model
        self.loss_metric_dict_train = dict.fromkeys(self._model.metrics_names)
        self.loss_metric_dict_valid = dict.fromkeys(self._model.metrics_names)
        for key_ in self._model.metrics_names:
            self.loss_metric_dict_train[key_] = list()
            self.loss_metric_dict_valid[key_] = list()
        self.loss_dict_init = True

    def _extract_data(self, data):
        nb_samples = len(data)
        obs = data[0]
        _x = []
        _sz_x = []
        _y = []
        _sz_y = []

        for attr_nm in self.attr_x:
            arr_ = getattr(obs, attr_nm)
            sz = arr_.size
            _sz_x.append(sz)
        
        for sz in _sz_x:
            _x.append(np.zeros((self.nb_samples, sz), dtype=self.dtype))

        for attr_nm in self.attr_y:
            arr_ = getattr(obs, attr_nm)
            sz = arr_.size
            _sz_y.append(sz)

        for sz in _sz_y:
            _y.append(np.zeros((nb_samples, sz), dtype=self.dtype))        

        for idx in range(nb_samples):
            for attr_nm, arr_ in zip(self.attr_x, _x):
                arr_[idx, :] = getattr(data[idx], attr_nm)
            for attr_nm, arr_ in zip(self.attr_y, _y):
                arr_[idx, :] = getattr(data[idx], attr_nm)

        return _x, _y
        

    def build_model(self):
        self._model = Sequential()
        inputs_x = [Input(shape=(el,), name="x_{}".format(nm_)) for el, nm_ in
                    zip(self._sz_x, self.attr_x)]

        # tensor_line_status = None
        if self._idx is not None:
            # line status is encoded: 1 disconnected, 0 connected
            # I invert it here
            if self._where_id == "x":
                self.tensor_line_status = inputs_x[self._idx]
            else:
                raise RuntimeError("Unknown \"where_id\"")
            self.tensor_line_status = 1.0 - self.tensor_line_status

        # encode each data type in initial layers
        encs_out = []
        for init_val, nm_ in zip(inputs_x, self.attr_x):
            lay = init_val

            for i, size in enumerate(self.sizes_enc):
                lay_fun = self._layer_fun(size,
                                          name="enc_{}_{}".format(nm_, i),
                                          activation=self._layer_act)
                lay = lay_fun(lay)
                if self._layer_act is None:
                    # add a non linearity if not added in the layer
                    lay = Activation("relu")(lay)
            encs_out.append(lay)

        # concatenate all that
        lay = tf.keras.layers.concatenate(encs_out)

        # i do a few layer
        for i, size in enumerate(self.sizes_main):
            lay_fun = self._layer_fun(size,
                                      name="main_{}".format(i),
                                      activation=self._layer_act)
            lay = lay_fun(lay)
            if self._layer_act is None:
                # add a non linearity if not added in the layer
                lay = Activation("relu")(lay)

        # i predict the full state of the grid given the input variables
        outputs_gm = []
        model_losses = {}

        for sz_out, nm_ in zip(self._sz_y,
                               self.attr_y):

            for i, size in enumerate(self.sizes_out):
                lay_fun = self._layer_fun(size,
                                          name="{}_{}".format(nm_, i),
                                          activation=self._layer_act)
                lay = lay_fun(lay)
                if self._layer_act is None:
                    # add a non linearity if not added in the layer
                    lay = Activation("relu")(lay)

            # predict now the variable
            name_output = "{}_hat".format(nm_)
            # force the model to output 0 when the powerline is disconnected
            if self.tensor_line_status is not None and nm_ in self._line_attr:
                pred_ = Dense(sz_out, name=f"{nm_}_force_disco")(lay)
                pred_ = tfk_multiply((pred_, self.tensor_line_status), name=name_output)
            else:
                pred_ = Dense(sz_out, name=name_output)(lay)

            outputs_gm.append(pred_)
            model_losses[name_output] = "mse"
            # model_losses.append(tf.keras.losses.mean_squared_error)

        # now create the model in keras
        self._model = Model(inputs=inputs_x,
                            outputs=outputs_gm,
                            name="model")
        # and "compile" it
        self._optimizer_model = optimizers.Adam(learning_rate=self.lr)
        self._model.compile(loss=model_losses, optimizer=self._optimizer_model)

    def train(self, training_data, epochs=1, batch_size=32, shuffle=False, save_model=True, tf_writer=None, verbose=1):
        """
        train the fully connected model on the batches of data
        """

        self.training_data = training_data

        x_preprocessed, y_preprocessed = self.preprocess_data(training_data)
        x_batches, y_batches, steps = self.get_batches(x_preprocessed, y_preprocessed, batch_size=batch_size, shuffle=shuffle)

        # whether to show the progress bar
        if verbose==1:
            pbar = tqdm(total=epochs*steps)

        # training and validation loop
        for epoch in range(epochs):
            train_losses_step = list()
            ################### training ##################
            for x_batch, y_batch in zip(x_batches, y_batches):
                train_batch_loss = self._train_model(x_batch, y_batch, tf_writer=tf_writer)
                train_losses_step.append(train_batch_loss)

                if verbose==1:
                    pbar.update(1)
            if not(self.loss_dicts_init):
                self.init_loss_dicts()

            # compute the mean of losses per epoch for training
            mean_epoch_train = np.mean(np.vstack(train_losses_step), axis=0)
            for key_, val_ in zip(self.loss_metric_dict_train.keys(), mean_epoch_train):
                self.loss_metric_dict_train[key_].append(val_)

            self.global_loss.append(mean_epoch_train[0])

            if verbose == 2:
                print("Epoch number {} / {} with loss {}".format(epoch,
                      epochs, self.loss_metric_dict_train["loss"][-1]))

            # TODO : add a validations steps
        
        if verbose==1:
            pbar.close()

        self.model_trained = True

        # TODO : here we should call the save methods of proxy        
        if save_model:
            pass
            #self.save_augmentedSimulator(path=self.model_path)

    def _train_model(self, x_batch, y_batch, tf_writer=None):
        if tf_writer is not None and self.__need_save_graph:
            tf.summary.trace_on()

        beg_ = time.time()
        batch_losses = self._model.train_on_batch(x=x_batch, y=y_batch)
        self._time_train += time.time() - beg_
        if tf_writer is not None and self.__need_save_graph:
            with tf_writer.as_default():
                tf.summary.trace_export("model-graph", 0)
            self.__need_save_graph = False
            tf.summary.trace_off()

        return batch_losses


    def predict(self, test_data, batch_size=1024, verbose=1, save_path=None):
        self.test_data = test_data
        x_preprocessed, y_preprocessed = self.preprocess_data(test_data)
        x_batches, y_batches, steps = self.get_batches(x_preprocessed, y_preprocessed, batch_size=batch_size, shuffle=False)
        self._y_hat = [[] for attr in self.attr_y]
        self._y = [[] for attr in self.attr_y]

        if verbose:
            pbar = tqdm(total=steps)

        # predicting over batches
        for x_batch, y_batch in zip(x_batches, y_batches):

            res = self._make_predictions(x_batch)

            for arr_, _pred in zip(self._y_hat, res):
                arr_.append(_pred)
            for arr_, _real in zip(self._y, y_batch):
                arr_.append(_real)
            
            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        self._y_hat = [np.asarray(np.vstack(el), dtype=np.float32) for el in self._y_hat]
        self._y = [np.asarray(np.vstack(el), dtype=np.float32) for el in self._y]



        # TODO : save the predictions
        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_predictions(save_path)

        return copy.deepcopy(self._y_hat), copy.deepcopy(self._y), self.predict_time

    def _model_predict(self, x_batch, training=False):
        beg_ = time.time()
        res = self._model(x_batch, training=training)
        self.predict_time += time.time() - beg_
        res = self._post_process(res)
        return res

    def data_to_dict(self):
        observations = dict()
        predictions = dict()

        for attr_nm in (*self.attr_x, *self.attr_y):
            observations[attr_nm] = np.vstack([getattr(obs, attr_nm) for obs in self.test_data])

        for attr_nm, var_ in zip(self.attr_y, self._y_hat):
            predictions[attr_nm] = var_

        return observations, predictions


    def save_predictions(self, path):
        observations, predictions = self.data_to_dict()

        if not os.path.exists(path):
            os.mkdir(path)
        simulator_path = os.path.join(path, self.name)
        if not os.path.exists(simulator_path):
            os.mkdir(simulator_path)

        for key_, val_ in predictions.items():
            np.save(os.path.join(simulator_path, f"{key_}_pred.npy"), val_)

        for key_, val_ in observations.items():
            np.save(os.path.join(simulator_path, f"{key_}_real.npy"), val_)

        print("Observations and Predictions are saved successfully !")


    def get_batches(self, x, y, batch_size=32, shuffle=False):
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
        steps = np.int(np.ceil((self.nb_samples) / batch_size))

        x_batches = list()
        y_batches = list()

        for step in range(steps-1):
            tmp_list = list()
            for arr_ in x:
                tmp_list.append(arr_[step*batch_size: (step+1)*batch_size])
            x_batches.append(tmp_list)

            tmp_list = list()
            for arr_ in y:
                tmp_list.append(arr_[step*batch_size: (step+1)*batch_size])
            y_batches.append(tmp_list)

        tmp_list = list()
        for arr_ in x:
            tmp_list.append(arr_[(step+1)*batch_size:])
        x_batches.append(tmp_list)

        tmp_list = list()
        for arr_ in y:
            tmp_list.append(arr_[(step+1)*batch_size:])
        y_batches.append(tmp_list)

        if shuffle:
            # TODO
            pass

        return x_batches, y_batches, steps
    
    def preprocess_data(self, data):
        """
        Maybe not here
        develop a similar function as _extract_data in LeapNet and BaseProxy
        
        Maybe develop it only in LeapNet Class because already exists in BaseProxy for x and y 
        whereas for x and y and tau is develpped in LeapNet
        """
        _x, _y = self._extract_data(data)
        tmpx = [(arr - m_) / sd_ for arr, m_, sd_ in zip(_x, self._m_x, self._sd_x)]
        tmpy = [(arr - m_) / sd_ for arr, m_, sd_ in zip(_y, self._m_y, self._sd_y)]
        
        tmpx = [tf.convert_to_tensor(el) for el in tmpx]
        tmpy = [tf.convert_to_tensor(el) for el in tmpy]
        return tmpx, tmpy

    def save(self, path, ext=".h5"):
        self._save_metadata(path)
        self.save_data(path=path, ext=ext)

    def load(self, path):
        """
        Load a previously stored experiments from the hard drive.

        This both load data for this class and from the proxy.

        Parameters
        ----------
        path: ``str``
            Where to load the experiment from.

        """
        if path is not None:
            if not os.path.exists(path):
                raise RuntimeError(f"You asked to load a model at \"{path}\" but there is nothing there.")

            self._load_metadata(path)
            self.build_model()
            self.load_data(path=path, ext=".h5")

    def _load_metadata(self, path_model):
        """load the metadata of the experiments (both for me and for the proxy)"""
        json_nm = "metadata.json"
        with open(os.path.join(path_model, json_nm), "r", encoding="utf-8") as f:
            me = json.load(f)
        self.load_metadata(me["proxy"])

    def get_metadata(self):
        res = super().get_metadata()
        # save attribute for the "extra" database
        res["attr_tau"] = [str(el) for el in self.attr_tau]
        res["_sz_tau"] = [int(el) for el in self._sz_tau]
        res["loss"] = self.loss

        # save means and standard deviation
        res["_m_x"] = []
        for el in self._m_x:
            self._save_dict(res["_m_x"], el)
        res["_m_y"] = []
        for el in self._m_y:
            self._save_dict(res["_m_y"], el)
        res["_m_tau"] = []
        for el in self._m_tau:
            self._save_dict(res["_m_tau"], el)
        res["_sd_x"] = []
        for el in self._sd_x:
            self._save_dict(res["_sd_x"], el)
        res["_sd_y"] = []
        for el in self._sd_y:
            self._save_dict(res["_sd_y"], el)
        res["_sd_tau"] = []
        for el in self._sd_tau:
            self._save_dict(res["_sd_tau"], el)

        # store the sizes
        res["sizes_enc"] = [int(el) for el in self.sizes_enc]
        res["sizes_main"] = [int(el) for el in self.sizes_main]
        res["sizes_out"] = [int(el) for el in self.sizes_out]

        return res

    def load_metadata(self, dict_):
        """
        load the metadata of this neural network (also called meta parameters) from a dictionary
        """
        self.attr_tau = tuple([str(el) for el in dict_["attr_tau"]])
        self._sz_tau = [int(el) for el in dict_["_sz_tau"]]
        self.loss = dict_["loss"]
        super().load_metadata(dict_)

        for key in ["_m_x", "_m_y", "_m_tau", "_sd_x", "_sd_y", "_sd_tau"]:
            setattr(self, key, [])
            for el in dict_[key]:
                self._add_attr(key, el)

        self.sizes_enc = [int(el) for el in dict_["sizes_enc"]]
        self.sizes_main = [int(el) for el in dict_["sizes_main"]]
        self.sizes_out = [int(el) for el in dict_["sizes_out"]]
        
        if "_layer_act" in dict_:
            self._layer_act = str(dict_["_layer_act"])
        else:
            self._layer_act = None


