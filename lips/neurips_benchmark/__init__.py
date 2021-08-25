# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import grid2op
from leap_net.agents import RandomRefAct1
from lightsim2grid import LightSimBackend


def get_kwargs_simulator_scenario1():
    """
    This function return the
    Returns
    -------

    """
    env_name = "l2rpn_case14_sandbox"

    # create a temporary environment to retrieve the default parameters of this specific environment
    env_tmp = grid2op.make(env_name)
    param = env_tmp.parameters
    param.NO_OVERFLOW_DISCONNECTION = True
    # i can act on all powerline / substation at once
    param.MAX_LINE_STATUS_CHANGED = 999999
    param.MAX_SUB_CHANGED = 999999
    # i can act every step on every line / substation (no cooldown)
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    return {"dataset": env_name,
            "param": param,
            "backend": LightSimBackend()}


def get_actor_training_scenario1(simulator):
    acts_dict_ = [{"set_bus": {"loads_id": [(4, 2)], "generators_id": [(2, 2)], "lines_or_id": [(9, 2), (8, 2)]}},
                  # sub_5_id1
                  {"set_bus": {"loads_id": [(3, 2)], "lines_or_id": [(17, 2)], "lines_ex_id": [(6, 2)]}},  # sub_4_id1
                  # {"set_bus": {"lines_or_id": [(17, 2)], "lines_ex_id": [(4, 2)]}},  # sub_4_id2
                  # {"set_bus": {"lines_or_id": [(15, 2)], "lines_ex_id": [(3, 2)]}},  # "sub3_id1"
                  # {"set_bus": {"lines_or_id": [(16, 2)], "lines_ex_id": [(3, 2)]}},  # "sub3_id2"
                  # {"set_bus": {"lines_or_id": [(6, 2), (16, 2)]}},  # "sub3_id3"
                  # {"set_bus": {"lines_or_id": [(15, 2)], "lines_ex_id": [(3, 2)], "loads_id": [(2, 2)]}},  # "sub3_id4"
                  # {"set_bus": {"lines_or_id": [(10, 2), (19, 2)], "loads_id": [(5, 2)]}},  # "sub8_id1"
                  # {"set_bus": {"lines_or_id": [(10, 2), (19, 2)]}},  # "sub8_id2"
                  # {"set_bus": {"lines_or_id": [(10, 2)], "loads_id": [(5, 2)]}},  # "sub8_id3"
                  # {"set_bus": {"lines_or_id": [(4, 2), (2, 2)]}},  # "sub1_id1" \n",
                  # {"set_bus": {"lines_or_id": [(3, 2)], "generators_id": [(0, 2)]}},  # "sub1_id2"
                  {"set_line_status": [(14, -1)]},  # "powerline_9_10"
                  {"set_line_status": [(12, -1)]},  # "powerline_12_13"
                  # composed action
                  {"set_bus": {"loads_id": [(4, 2), (3, 2)],
                               "generators_id": [(2, 2)],
                               "lines_or_id": [(9, 2), (8, 2), (17, 2)],
                               "lines_ex_id": [(6, 2)]}},
                  ]
    agent_seed = 1
    env = simulator._simulator
    li_act = [env.action_space(el) for el in acts_dict_]
    agent = RandomRefAct1(env.action_space, p=0.5, list_act=li_act)
    return agent
