# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import numpy as np
import grid2op
from grid2op.Agent import BaseAgent

class AgentTC1(BaseAgent):
    """
    The Topological Change One (TC1) agent will randomly switch the buses of one of the indicated elements. 
    Only one substation is affected by the topological action returned after each call to the act function.
    Contrary to RandomTC1, it does not reconnect the elements which are not affected by the topological change
    to the first bus !

    params
    ______
        action_space : Grid2op ``Environment`` ``action_space``

        subsid_list : ``list``
            a list of substation identifiers for which we would perform topological changes


    """
    def __init__(self, action_space, subsid_list):
        super(AgentTC1, self).__init__(action_space)
        if not "set_bus" in action_space.subtype.authorized_keys:
            raise NotImplementedError("Impossible to have a TopoChange agent if you cannot set the bus")
        #self.subsid_list = [1,3,4,5,8,12] 
        
        self.act_space = grid2op.Converter.IdToAct(action_space)
        # generate only "set_topo" type of actions
        self.act_space.init_converter(set_line_status=False, change_line_status=False, change_bus_vect=False, redispatch=False)
        
        # filter only the action at the relevant substations
        def filter_fun(g2op_act):
            l_imp, s_imp = g2op_act.get_topological_impact()
            res = False
            if np.any(s_imp[subsid_list]):
                res = True
            return res
        
        self.act_space.filter_action(filter_fun)

        self.bus_switch_action_space_as_list = []
        # generate bus switching actions that impact only one substation.
        for act_ in self.act_space.all_actions:
            self.bus_switch_action_space_as_list.append(act_)
        self.actions_size = len(self.bus_switch_action_space_as_list)

    def act(self, obs, reward, done):
        # select and return randomly an action (at one substation) in the topological action list
        id_ = self.space_prng.choice(self.actions_size)
        # adding the reset action with an action which will change the buses at one substation
        return self.bus_switch_action_space_as_list[id_]