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


class RandomN2(BaseAgent):
    """
    This "agent" will randomly disconnect exactly 1 powerline from the grid.

    **NB** Every powerline that is not chosen at random to be disconnected will be reconnected by force.

    This agent will modify the status of all powerlines at every steps!
    """

    def __init__(self, action_space):
        super(RandomN2, self).__init__(action_space)
        if not "set_line_status" in action_space.subtype.authorized_keys:
            raise NotImplementedError("Impossible to have a RandomN1 agent if you cannot set the status or powerline")

        # represent the action "exactly one powerline is disconnected
        self.powerline_actions = np.ones(action_space.n_line, dtype=dt_int)
        self.tmp_arr = np.ones(action_space.n_line, dtype=dt_int)

    def act(self, obs, reward, done):
        id_1 = self.space_prng.choice(self.action_space.n_line)
        id_2 = self.space_prng.choice(self.action_space.n_line - 1)
        if id_2 >= id_1:
            # this procedure is to be sure not to "disconnect twice" the same powerline
            id_2 += 1
        self.tmp_arr[:] = self.powerline_actions
        self.tmp_arr[id_1] = -1
        self.tmp_arr[id_2] = -1
        li_bus = [(i, el) for i, el in enumerate(self.tmp_arr)]
        act = self.action_space({"set_line_status": self.tmp_arr,
                                 "set_bus": {"lines_or_id": li_bus,
                                             "lines_ex_id": li_bus},
                                 }
                                )
        return act
