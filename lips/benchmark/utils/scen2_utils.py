# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""
Set of utility function to generate the data for Benchmark2
"""
from grid2op import Agent
from .actor_utils import ChangeTopoRefN1, ChangeTopoRefN2, ChangeTopoRefN1Ref

# for reference
REF_ACTION = [# sub_5_id1
              {"set_bus": {"loads_id": [(4, 2)], "generators_id": [(2, 2)], "lines_or_id": [(9, 2), (8, 2)]}},
              # sub_4_id1
              {"set_bus": {"loads_id": [(3, 2)], "lines_or_id": [(17, 2)], "lines_ex_id": [(6, 2)]}},
              {"set_bus": {"lines_or_id": [(17, 2)], "lines_ex_id": [(4, 2)]}},  # sub_4_id2
              {"set_bus": {"lines_or_id": [(15, 2)], "lines_ex_id": [(3, 2)]}},  # "sub3_id1"
              {"set_bus": {"lines_or_id": [(16, 2)], "lines_ex_id": [(3, 2)]}},  # "sub3_id2"
              {"set_bus": {"lines_or_id": [(6, 2), (16, 2)]}},  # "sub3_id3"
              {"set_bus": {"lines_or_id": [(15, 2)], "lines_ex_id": [(3, 2)], "loads_id": [(2, 2)]}},  # "sub3_id4"
              # "sub8_id1"
              {"set_bus": {"lines_or_id": [(10, 2), (19, 2)], "loads_id": [(5, 2)]}},
              # "sub8_id2"
              {"set_bus": {"lines_or_id": [(10, 2), (19, 2)]}},
              # "sub8_id3"
              {"set_bus": {"lines_or_id": [(10, 2)], "loads_id": [(5, 2)]}},
              {"set_bus": {"lines_or_id": [(4, 2), (2, 2)]}},  # "sub1_id1" \n",
              {"set_bus": {"lines_or_id": [(3, 2)], "generators_id": [(0, 2)]}},  # "sub1_id2"
              {"set_line_status": [(14, -1)]},  # "powerline_9_10"
              {"set_line_status": [(12, -1)]},  # "powerline_12_13"
              {"set_line_status": [(3, -1)]},  # "powerline_1_3"
              # composed action sub_5_id1 + sub_4_id1
              {"set_bus": {"loads_id": [(4, 2), (3, 2)],
                           "generators_id": [(2, 2)],
                           "lines_or_id": [(9, 2), (8, 2), (17, 2)],
                           "lines_ex_id": [(6, 2)]}},
              # composed action sub_5_id1 + sub8_id1
              {"set_bus": {"loads_id": [(4, 2), (5, 2)],
                           "generators_id": [(2, 2)],
                           "lines_or_id": [(9, 2), (8, 2), (10, 2), (19, 2)],
                           "lines_ex_id": []}},
              # composed action sub_4_id1 + sub8_id1
              {"set_bus": {"loads_id": [(3, 2), (5, 2)],
                           "lines_or_id": [(17, 2), (10, 2), (19, 2)],
                           "lines_ex_id": [(6, 2)]}},
              # composed action sub_1_id1 + sub_3_id4
              {"set_bus": {"loads_id": [(2, 2)],
                           "lines_or_id": [(15, 2), (4, 2), (2, 2)],
                           "lines_ex_id": [(3, 2)]}},
              ]

def _aux_act_scenario(env) -> tuple:
    """Auxiliary function to prepare required actions to take

    It returns singular topo actions (acting on one sub)

    Parameters
    ----------
    env : ``grid2op.Environment``
        the environment on which actions should be taken

    Returns
    -------
    ``tuple``
        list of topological action and list of N-1 actions
    """
    li_ref_topo = [{"set_line_status": [(l_id, -1)]} for l_id in range(env.n_line)]
    li_act_n1 = [REF_ACTION[0], REF_ACTION[1], REF_ACTION[6], REF_ACTION[7], REF_ACTION[10]]
    li_ref_topo = [env.action_space(el) for el in li_ref_topo]
    li_act_n1 = [env.action_space(el) for el in li_act_n1]
    return li_ref_topo, li_act_n1

def _aux_act_scenario_ood(env) -> tuple:
    """Auxiliary function to prepare required actions to take

    It returns composed topo actions (acting on two subs) to test the out-of-distribution

    Parameters
    ----------
    env : ``grid2op.Environment``
        the environment on which actions should be taken

    Returns
    -------
    ``tuple``
        list of topological action and list of N-1 actions
    """
    li_ref_topo = [{"set_line_status": [(l_id, -1)]} for l_id in range(env.n_line)]
    li_act_n1 = [REF_ACTION[15], REF_ACTION[16], REF_ACTION[17], REF_ACTION[18]]
    li_ref_topo = [env.action_space(el) for el in li_ref_topo]
    li_act_n1 = [env.action_space(el) for el in li_act_n1]
    return li_ref_topo, li_act_n1

def get_actor_training_scenario(simulator) -> Agent:
    """Gets actor for training scenario

    Parameters
    ----------
    simulator : PhysicalSimulator
        The simulator instance

    Returns
    -------
    ``grid2op.Agent``
        the agent which acts on the environment

    Todo
    ----
    TODO : simulator could be replaced by config
    """
    env = simulator._simulator
    li_ref_topo, li_act_n1 = _aux_act_scenario(env)
    agent = ChangeTopoRefN1Ref(env.action_space,
                               p=0.5,
                               ref_topo=li_ref_topo,
                               list_act=li_act_n1)
    return agent


def get_actor_test_scenario(simulator) -> Agent:
    """Get actor for test scenario

    Parameters
    ----------
    simulator : PhysicalSimulator
        The simulator instance

    Returns
    -------
    ``grid2op.Agent``
        the agent which acts on the environment
    """
    env = simulator._simulator
    li_ref_topo, li_act_n1 = _aux_act_scenario(env)
    agent = ChangeTopoRefN1(env.action_space,
                            ref_topo=li_ref_topo,
                            list_act=li_act_n1)
    return agent


def get_actor_test_ood_topo_scenario(simulator) -> Agent:
    """Get actor for test ood scenario

    Parameters
    ----------
    simulator : PhysicalSimulator
        The simulator instance

    Returns
    -------
    ``grid2op.Agent``
        the agent which acts on the environment
    """
    env = simulator._simulator
    li_ref_topo, li_act_n1 = _aux_act_scenario_ood(env)
    agent = ChangeTopoRefN1(env.action_space,
                            ref_topo=li_ref_topo,
                            list_act=li_act_n1)
    return agent
