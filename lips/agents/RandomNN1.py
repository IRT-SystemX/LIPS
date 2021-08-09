# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.dtypes import dt_int, dt_float
import pdb


class RandomNN1(BaseAgent):
    """
    This "agent" will randomly disconnect 0 or 1 powerline with probability 1-p to disconnect 1 powerline
    and with probability p to disconnect randomly one powerline.
    
    **NB** the output distribution is heavily biased: all the powerline are connected with proba `1-p`, but if you
    consider the odds of having powerline line `l` disconnected it's equal to `p / (nb_line)`.
    """
    def __init__(self, action_space, p):
        super(RandomNN1, self).__init__(action_space)
        if not "set_line_status" in action_space.subtype.authorized_keys:
            raise NotImplementedError("Impossible to have a RandomN1 agent if you cannot set the status or powerline")
        if p <= 0.:
            raise RuntimeError("Impossible to have p lower than 0.")
        self.p = dt_float(p)
        self._1_p = 1. - self.p
        # represent the action "no powerline are disconnected
        self.reset_all = np.ones(action_space.n_line, dtype=dt_int)
        # represent the action "exactly one powerline is disconnected
        self.powerline_actions = 1 - 2*np.eye(action_space.n_line, dtype=dt_int)

    def act(self, obs, reward, done):
        ur = self.space_prng.uniform()
        if ur < self._1_p:
            arr_ = self.reset_all
        else:
            id_ = self.space_prng.choice(self.action_space.n_line)
            arr_ = self.powerline_actions[id_, :]
        li_bus = [(i, el) for i, el in enumerate(arr_)]
        act = self.action_space({"set_line_status": arr_,
                                 "set_bus": {"lines_or_id": li_bus,
                                             "lines_ex_id": li_bus},
                                 })
        return act
