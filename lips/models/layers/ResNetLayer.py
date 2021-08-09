# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add as tfk_add


import tensorflow as tf

import pdb


class ResNetLayer(Layer):
    """
    This layer implements a "ResNet block". If inputs are `x` the "resnet block" here produce the outputs:
    `y` = `x` + `act( Dense(dim_x)(act(Dense(layer_size)(x))) )`

    This is experimental, and any usage of another resnet implementation is probably better suited than this one.

    """
    def __init__(self,
                 units,
                 initializer='glorot_uniform',
                 use_bias=True,
                 trainable=True,
                 name=None,
                 activation="linear",
                 **kwargs):
        super(ResNetLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.initializer = initializer
        self.use_bias = use_bias
        self.units = int(units)
        self.activation = activation

        self.e = None
        self.d = None

    def build(self, input_shape):
        nm_e = "e"
        nm_d = "d"
        if self.name is not None:
            nm_e = '{}_e'.format(self.name)
            nm_d = '{}_d'.format(self.name)

        self.e = Dense(self.units,
                       kernel_initializer=self.initializer,
                       use_bias=self.use_bias,
                       trainable=self.trainable,
                       name=nm_e)
        self.d = Dense(input_shape[-1],
                       kernel_initializer=self.initializer,
                       use_bias=self.use_bias,
                       trainable=self.trainable,
                       name=nm_d)

    def get_config(self):
        config = super(ResNetLayer, self).get_config().copy()
        config.update({
            'initializer': str(self.initializer),
            'use_bias': bool(self.use_bias),
            "units": int(self.units),
            "activation": str(self.activation)
        })
        return config

    def call(self, inputs, **kwargs):
        tmp = self.e(inputs)
        if self.activation is not None:
            tmp = Activation(self.activation)(tmp)
        tmp = self.d(tmp)
        if self.activation is not None:
            tmp = Activation(self.activation)(tmp)
        res = tfk_add([inputs, tmp])
        return res
