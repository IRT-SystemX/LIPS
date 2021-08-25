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
import tensorflow as tf
import tensorflow.keras.optimizers as tfko
from tensorflow.keras.layers import Dense

from lips.simulators import BaseNNProxy


class AugmentedSimulator(BaseNNProxy):
    """
    AugmentedSimulator is an adaptation of BaseNNProxy from leap_net library and offers various functionality to train and 
    to perform inference
    """
    # TODO : mayber modify some of arguments or add some if required with new functionalities
    def __init__(self,
                 name,
                 lr=1e-4,
                 train_batch_size=32,
                 eval_batch_size=1024,
                 layer=Dense,
                 layer_act=None,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),  # input that will be given to the proxy
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),  # output that we want the proxy to predict
                ):
        # TODO : mayber modify some of arguments or add some if required with new functionalities
        BaseNNProxy.__init__(self,
                             name=name,
                             lr=lr,
                             train_batch_size=train_batch_size,
                             eval_batch_size=eval_batch_size,
                             layer=layer,
                             layer_act=layer_act,
                             attr_x=attr_x,
                             attr_y=attr_y                             
                            )
        
        self._batch_x_train = None
        self._batch_x_valid = None
        self._batch_y_train = None
        self._batch_y_valid = None
        
    def train(self, tf_writer=None):
        """
        override the train function of BaseNNProxy
        
        this function could also do the validation step of learning
        """
        # use _train_model of BaseNNProxy for train on batch in this function
        
        # preprocess the batch before training
        batch = self.preprocess_batch()
        
        if tf_writer is not None and self.__need_save_graph:
            tf.summary.trace_on()

        beg_ = time.time()
        batch_losses = self._train_model(batch)
        self._time_train += time.time() - beg_
        if tf_writer is not None and self.__need_save_graph:
            with tf_writer.as_default():
                tf.summary.trace_export("model-graph", 0)
            self.__need_save_graph = False
            tf.summary.trace_off()
        return batch_losses
    

    def predict(self):
        """
        override this function which is in BaseProxy which it is not recommended 
        """
        # it should call _make_predictions(data) of BaseNNPorxy
        
        # preprocess the batch before performing inference
        batch = self.preprocess_batch()
        
        beg_ = time.time()
        res = self._make_predictions(batch, training=False)
        self._time_predict += time.time() - beg_
        res = self._post_process(res)
        return res
        
    
    """
    def _get_adds_mults_from_name(self):
        
        #override the _get_adds_mults_from_name of BaseProxy to adapt it to new data format
        # maybe not with new method : obss variable is created during train data generation to init the proxy
        pass
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
    
    
    def store_batch(self, batch):
        """
        to store the batches in local variables
        """
        self._batch_x_train = list()
        self._batch_y_train = list()
        
        for attr_nm in self.attr_x:
            self._batch_x_train.append(batch.get(attr_nm))
            
        for attr_nm in self.attr_y:
            self._batch_y_train.append(batch.get(attr_nm))
