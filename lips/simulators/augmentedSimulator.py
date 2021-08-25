# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import numpy as np
import copy
import json

from lips.simulators import Simulator
from lips.simulators import BaseNNProxy


class AugmentedSimulator(Simulator, BaseNNProxy):
    """
    AugmentedSimulator is an adaptation of BaseNNProxy from leap_net library and offers various functionality to train and 
    to perform inference
    """
    # TODO : mayber modify some of arguments or add some if required with new functionalities
    def __init__(self,
                 name="augmented_simulator"
                ):
        # TODO : mayber modify some of arguments or add some if required with new functionalities
        Simulator.__init__(self, name=name)
        BaseNNProxy.__init__(self, name=name)
        self.nb_samples = None
        self.obs_list = None

    def init(self,obs_list):
        self.nb_samples = len(obs_list)
        self.obs_list = obs_list
        self.process_obs(obs_list)

        # get the mean and standard deviation for normalization
        if not self._metadata_loaded:
            # for the input
            self._m_x = []
            self._sd_x = []
            for attr_nm in self.attr_x:
                self._m_x.append(self._get_mean(obs_list, attr_nm))
                self._sd_x.append(self._get_sd(obs_list, attr_nm))

            # for the output
            self._m_y = []
            self._sd_y = []
            for attr_nm in self.attr_y:
                self._m_y.append(self._get_mean(obs_list, attr_nm))
                self._sd_y.append(self._get_sd(obs_list, attr_nm))

        self._metadata_loaded = True

    def process_obs(self, obs_list):
        obs = obs_list[0]
        self._x = []
        self._sz_x = []
        self._y = []
        self._sz_y = []

        for attr_nm in self.attr_x:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_x.append(sz)
        
        for sz in self._sz_x:
            self._x.append(np.zeros((self.nb_samples, sz), dtype=self.dtype))

        for attr_nm in self.attr_y:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_y.append(sz)

        for sz in self._sz_y:
            self._y.append(np.zeros((self.nb_samples, sz), dtype=self.dtype))        

        for idx in range(self.nb_samples):
            for attr_nm, arr_ in zip(self.attr_x, self._x):
                arr_[idx, :] = getattr(obs_list[idx], attr_nm)
            for attr_nm, arr_ in zip(self.attr_y, self._y):
                arr_[idx, :] = getattr(obs_list[idx], attr_nm)

        return self._x, self._y

    def build_model(self):
        """
        a tensorflow or pytorch model could be built here
        """
        pass
        
    def train(self, tf_writer=None):
        """
        override the train function of BaseNNProxy
        
        this function could also do the validation step of learning

        It should be used to train the model using training data
        """
        pass
    

    def predict(self):
        """
        override this function which is in BaseProxy 

        this function should be used to predict using test data
        """
        pass

    def save(self):
        """
        This function allows to save the model weights
        """
        pass

    def load(self):
        """
        this function allows to load the model weights and build it
        """
        
    def _save_metadata(self, path_model):
        """
        save the dimensions of the models and the scalers
        the same as in AgentWithProxy to save the proxy metadata
        """
        # it should call all the get_metadata function of all the sub classes to gather all the information
        # note that a get_metadata is also available in outer class ProxyLeapNet (how to access it ?)
        # if it is possible, gather all these data in a new get_metadata overriden in this class
        json_nm = "metadata.json"
        res = dict()
        res["proxy"] = self.get_metadata()
        with open(os.path.join(path_model, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=res, fp=f)

    def _load_metadata(self, load_path):
        pass
