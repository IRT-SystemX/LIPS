# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import copy
import warnings
import json
import os

import tensorflow as tf
import tensorflow.keras.optimizers as tfko
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import multiply as tfk_multiply


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Input


from leap_net.LtauNoAdd import LtauNoAdd

from lips.simulators import ProxyLeapNet
from lips.simulators import AugmentedSimulator


class LeapNet(ProxyLeapNet, AugmentedSimulator):
    """
    # TODO merge the two concept better ! And respect new API

    This class inherit from both ProxyLeapNet and AugmentedSimulator
    
    ``ProxyLeapNet`` class is an implementation of LeapNet model which is available in leap_net library
    
    ``AugmentedSimulator`` is an adaptation of BaseNNProxy of leap_net library for LIPS benchmark platform
    it offers different functionalities to train a Neural network based model and to perform inference
    to save and to load the augmented simulator and its specific meta data
    """
    # TODO : mayber modify some of arguments or add some if required with new functionalities
    def __init__(self,
                 name="leap_net",
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 attr_tau=("line_status",),
                 sizes_enc=(20, 20, 20),
                 sizes_main=(150, 150, 150),
                 sizes_out=(100, 40),
                 lr=1e-4,
                 scale_main_layer=None,  # increase the size of the main layer
                 scale_input_dec_layer=None,  # scale the input of the decoder
                 scale_input_enc_layer=None,  # scale the input of the encoder
                 layer=Dense,  # TODO (for save and restore)
                 layer_act=None,
                 loss="mse",
                 topo_vect_to_tau="raw",  # see code for now
                 kwargs_tau=None,  # optionnal kwargs depending on the method chosen for building tau from the observation
                 ):
        AugmentedSimulator.__init__(self,
                                    name=name,
                                    lr=lr,
                                    attr_x=attr_x,
                                    attr_y=attr_y,
                                    layer=layer,
                                    layer_act=layer_act)
        
        ProxyLeapNet.__init__(self,
                              name=name,
                              attr_x=attr_x,
                              attr_y=attr_y,
                              attr_tau=attr_tau,
                              sizes_enc=sizes_enc,
                              sizes_main=sizes_main,
                              sizes_out=sizes_out,
                              lr=lr,
                              scale_main_layer=scale_main_layer,  # increase the size of the main layer
                              scale_input_dec_layer=scale_input_dec_layer,  # scale the input of the decoder
                              scale_input_enc_layer=scale_input_enc_layer,  # scale the input of the encoder
                              layer=layer,  # TODO (for save and restore)
                              layer_act=layer_act,
                              topo_vect_to_tau=topo_vect_to_tau,  # see code for now
                              kwargs_tau=kwargs_tau,  # optionnal kwargs depending on the method chosen for building tau from the observation
                              )
        self._batch_tau_train = None
        self._batch_tau_valid = None
        
        self.loss = loss
    
    def build_model(self):
        """
        overriding the ProxyLeapNet build model to be able to customize it
        """
        if self._model is not None:
            # model is already initialized
            return
        self._model = Sequential()
        inputs_x = [Input(shape=(el,), name="x_{}".format(nm_)) for el, nm_ in
                    zip(self._sz_x, self.attr_x)]
        inputs_tau = [Input(shape=(el,), name="tau_{}".format(nm_)) for el, nm_ in
                      zip(self._sz_tau, self.attr_tau)]

        # tensor_line_status = None
        if self._idx is not None:
            # line status is encoded: 1 disconnected, 0 connected
            # I invert it here
            if self._where_id == "x":
                self.tensor_line_status = inputs_x[self._idx]
            elif self._where_id == "tau":
                self.tensor_line_status = inputs_tau[self._idx]
            else:
                raise RuntimeError("Unknown \"where_id\"")
            self.tensor_line_status = 1.0 - self.tensor_line_status

        # encode each data type in initial layers
        encs_out = []
        for init_val, nm_ in zip(inputs_x, self.attr_x):
            lay = init_val

            if self._scale_input_enc_layer is not None:
                # scale up to have higher dimension
                lay = Dense(self._scale_input_enc_layer,
                            name=f"scaling_input_encoder_{nm_}")(lay)
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

        if self._scale_main_layer is not None:
            # scale up to have higher dimension
            lay = Dense(self._scale_main_layer, name="scaling_inputs")(lay)

        # i do a few layer
        for i, size in enumerate(self.sizes_main):
            lay_fun = self._layer_fun(size,
                                      name="main_{}".format(i),
                                      activation=self._layer_act)
            lay = lay_fun(lay)
            if self._layer_act is None:
                # add a non linearity if not added in the layer
                lay = Activation("relu")(lay)

        # now i do the leap net to encode the state
        encoded_state = lay
        for input_tau, nm_ in zip(inputs_tau, self.attr_tau):
            tmp = LtauNoAdd(name=f"leap_{nm_}")([lay, input_tau])
            encoded_state = tf.keras.layers.add([encoded_state, tmp], name=f"adding_{nm_}")

        # i predict the full state of the grid given the input variables
        outputs_gm = []
        model_losses = {}
        # model_losses = []
        lossWeights = {}  # TODO
        for sz_out, nm_ in zip(self._sz_y,
                               self.attr_y):
            lay = encoded_state
            if self._scale_input_dec_layer is not None:
                # scale up to have higher dimension
                lay = Dense(self._scale_input_dec_layer,
                            name=f"scaling_input_decoder_{nm_}")(lay)
                lay = Activation("relu")(lay)

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
            model_losses[name_output] = self.loss
            # model_losses.append(tf.keras.losses.mean_squared_error)

        # now create the model in keras
        self._model = Model(inputs=(inputs_x, inputs_tau),
                            outputs=outputs_gm,
                            name="model")
        # and "compile" it
        self._schedule_lr_model, self._optimizer_model = self._make_optimiser()
        self._model.compile(loss=model_losses, optimizer=self._optimizer_model)
    
    def preprocess_batch(self):
        """
        Maybe not here
        develop a similar function as _extract_data in LeapNet and BaseProxy
        
        Maybe develop it only in LeapNet Class because already exists in BaseProxy for x and y 
        whereas for x and y and tau is develpped in LeapNet
        """
        
        tmpx = [(arr - m_) / sd_ for arr, m_, sd_ in zip(self._batch_x_train, self._m_x, self._sd_x)]
        tmpt = [(arr - m_) / sd_ for arr, m_, sd_ in zip(self._batch_tau_train, self._m_tau, self._sd_tau)]
        tmpy = [(arr - m_) / sd_ for arr, m_, sd_ in zip(self._batch_y_train, self._m_y, self._sd_y)]
        
        tmpx = [tf.convert_to_tensor(el) for el in tmpx]
        tmpt = [tf.convert_to_tensor(el) for el in tmpt]
        tmpy = [tf.convert_to_tensor(el) for el in tmpy]
        return (tmpx, tmpt), tmpy
    
    def store_batch(self, batch):
        """
        to store the batches in local variables
        
        batch will be a dictioanry in this case which can contain training and validation batches
        or simply a list of lists    
        batch {
            "prod_p", "prod_v", "load_p", "line_status", "p_or", "a_or" , etc.
        }
        """
        # suppose that batch is a dictionary
        self._batch_tau_train = list()
        for attr_nm in self.attr_tau:
            self._batch_tau_train.append(batch.get(attr_nm))

        AugmentedSimulator.store_batch(self, batch)   

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

        # store some information about some transformations we can do
        if self._scale_main_layer is not None:
            res["_scale_main_layer"] = int(self._scale_main_layer)
        else:
            # i don't store anything if it's None
            pass
        if self._scale_input_dec_layer is not None:
            res["_scale_input_dec_layer"] = int(self._scale_input_dec_layer)
        else:
            # i don't store anything if it's None
            pass
        if self._scale_input_enc_layer is not None:
            res["_scale_input_enc_layer"] = int(self._scale_input_enc_layer)
        else:
            # i don't store anything if it's None
            pass
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
        if "_scale_main_layer" in dict_:
            self._scale_main_layer = int(dict_["_scale_main_layer"])
        else:
            self._scale_main_layer = None
        if "_scale_input_dec_layer" in dict_:
            self._scale_input_dec_layer = int(dict_["_scale_input_dec_layer"])
        else:
            self._scale_input_dec_layer = None
        if "_scale_input_enc_layer" in dict_:
            self._scale_input_enc_layer = int(dict_["_scale_input_enc_layer"])
        else:
            self._scale_input_enc_layer = None
        if "_layer_act" in dict_:
            self._layer_act = str(dict_["_layer_act"])
        else:
            self._layer_act = None
