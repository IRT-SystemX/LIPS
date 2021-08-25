# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import numpy as np
from leap_net.proxy import BaseProxy

class Simulator(BaseProxy):
    """
    It is the base class, on the basis of which a physical or an augmented Simulator could be constructed
    """
    def __init__(self,
                 name
                ):

        BaseProxy.__init__(self, name=name)

        # inputs and outputs and their respective sizes
        self._x = None
        self._sz_x = None
        self._y = None
        self._sz_y = None
        self._y_hat = None


    def train(self, dataset):
        """
        This function is only used by an augmented simulator to train the model 
        """
        pass

    def predict(self, dataset):
        """
        This function could be called by a physical simulator or an augmented one to predict the results 
        """
        pass

    def save(self):
        """
        to save or restore the model weights
        """
        pass

    def save_metadata(self):
        """
        to save or restore the meta data
        """
        pass

    def process_obs(self):
        """
        function that process a grid2op observation (returned by the dataset) and computes the `x`, `y` (and `tau` if required)
        required for a physical or for an augmented simulator

        This function process only the observation to x and y, the process_obs in proxyLeapNet will process the tau if required
        """
        pass

    def init(self, obss):
        """
        To intialize the model from the dataset
        For an augmented simulator, this function will standardize the generated dataset
        """
        pass

   



