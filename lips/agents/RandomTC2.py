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

class RandomTC2(BaseAgent):
    """
    This agent will randomly switch the buses of some indicated elements. 
    Two substations will be affected by the action returned after each call to the act function.
    """
    def __init__(self, action_space, subsid_list):
        super(RandomTC2, self).__init__(action_space)
        if not "set_bus" in action_space.subtype.authorized_keys:
            raise NotImplementedError("Impossible to have a TopoChange agent if you cannot set the bus vector")
        #self.subsid_list = [1,3,4,5,8,12] 
        
        # reset all the connections and connect all the elements to bus 1
        self.set_buses = action_space({"set_bus": [(i,1) for i in range(action_space.dim_topo)]})
        
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
        
        for act1 in self.act_space.all_actions:
            _, s_imp1 = act1.get_topological_impact()
            s_id1 = np.where(s_imp1)[0]
            if not s_id1:
                # this is the do nothing action
                continue
            for act2 in self.act_space.all_actions:
                _, s_imp2 = act2.get_topological_impact()
                s_id2 = np.where(s_imp2)[0]
                if not s_id2:
                    # this is the do nothing action
                    continue
                if s_id2 <= s_id1: # To avoid duplicate actions.
                    continue
    
                act_saved = action_space()
                act_saved += act1
                act_saved += act2
                self.bus_switch_action_space_as_list.append(act_saved)
        self.actions_size = len(self.bus_switch_action_space_as_list)

    def act(self, obs, reward, done):
        # select and return randomly an action (at two substations) in the topological action list
        id_ = self.space_prng.choice(self.actions_size)
        return self.set_buses + self.bus_switch_action_space_as_list[id_]