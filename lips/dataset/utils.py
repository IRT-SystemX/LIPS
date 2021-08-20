# Copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import warnings
import numpy as np

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Parameters import Parameters
from grid2op.Rules import AlwaysLegal

from lips.agents import *


def get_parameters():
    # generate the environments parameters
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    param.MAX_SUB_CHANGED = 99999
    param.MAX_LINE_STATUS_CHANGED = 99999
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    return param


def create_env(env_name, use_lightsim_if_available=True):
    """create the grid2op environment with the right parameters and chronics class"""
    backend_cls = None
    if use_lightsim_if_available:
        try:
            from lightsim2grid.LightSimBackend import LightSimBackend
            backend_cls = LightSimBackend
        except ImportError as exc_:
            warnings.warn(
                "You ask to use lightsim backend if it's available. But it's not available on your system.")

    if backend_cls is None:
        from grid2op.Backend import PandaPowerBackend
        backend_cls = PandaPowerBackend

    param = get_parameters()

    env = grid2op.make(env_name,
                       param=param,
                       backend=backend_cls(),
                       gamerules_class=AlwaysLegal,
                       chronics_class=MultifolderWithCache
                       )
    return env


def reproducible_exp(env, agent, env_seed=None, chron_id_start=None, agent_seed=None):
    """
    ensure the reproducibility for the data, but NOT for tensorflow

    the environment need to be reset after a call to this method
    """
    if env_seed is not None:
        env.seed(env_seed)

    if chron_id_start is not None:
        # set the environment to start at the right chronics
        env.chronics_handler.tell_id(chron_id_start - 1)

    if agent_seed is not None:
        agent.seed(agent_seed)


def get_agent(env, agent_name, agent_params):
    if agent_name == "do_nothing":
        res = DoNothingAgent(env.action_space)
    elif agent_name == "random_nn1":
        res = RandomNN1(env.action_space, agent_params["p"])
    elif agent_name == "random_n1":
        res = RandomN1(env.action_space)
    elif agent_name == "random_n2":
        res = RandomN2(env.action_space)
    elif agent_name == "random_tc_nn1":
        res = RandomTCN1(env.action_space,
                         agent_params["subsid_list"], agent_params["p"])
    elif agent_name == "random_tc_n1":
        res = RandomTC1(env.action_space, agent_params["subsid_list"])
    elif agent_name == "random_tc_n2":
        res = RandomTC2(env.action_space, agent_params["subsid_list"])
    else:
        raise NotImplementedError()
    return res


def get_reference_action(env, reference_number=None, seed=14):
    """
    Get the maipulation of environment corresponding to each benchmark
    """
    if (reference_number is None) | (reference_number == 0):
        # default connected power network with no change
        return env.action_space()
    elif reference_number == 1:
        # Disconnecting the line 1_3_3
        # return env.action_space.disconnect_powerline(line_name="1_3_3", previous_action=action)
        return env.action_space({"set_line_status": [("1_3_3", -1)]})
    elif reference_number == 2:
        # Topology change at node 3
        agent = RandomTC1(env.action_space, subsid_list=[3])
        agent.seed(seed)
        return agent.act(None, None, None)
    elif reference_number == 3:
        # Topology change at node 5
        agent = RandomTC1(env.action_space, subsid_list=[5])
        agent.seed(seed)
        return agent.act(None, None, None)
    elif reference_number == 4:
        # Topology changes at nodes 3 and 5
        agent = RandomTC2(env.action_space, subsid_list=[3, 5])
        agent.seed(seed)
        return agent.act(None, None, None)
    else:
        raise NotImplementedError()


def apply_reference_action(env, action=None, reference_number=None, seed=14):

    if (reference_number is None) | (reference_number == 0):
        # default connected power network with no change
        return action
    elif reference_number == 1:
        # Disconnecting the line 1_3_3
        return disconnet_power_line(action, line_id=3, line_name="1_3_3")
    elif reference_number == 2:
        # Topology change at node 3
        agent = AgentTC1(env.action_space, subsid_list=[3])
        agent.seed(seed)
        act_ref = agent.act(None, None, None)
        return action + act_ref
    elif reference_number == 3:
        # Topology change at node 5
        agent = AgentTC1(env.action_space, subsid_list=[5])
        agent.seed(seed)
        act_ref = agent.act(None, None, None)
        return action + act_ref
    elif reference_number == 4:
        # Topology changes at nodes 3 and 5
        agent = AgentTC2(env.action_space, subsid_list=[3, 5])
        agent.seed(seed)
        act_ref = agent.act(None, None, None)
        return action + act_ref
    else:
        raise NotImplementedError()


def disconnet_power_line(action, line_id, line_name):
    action.line_or_set_bus = [(line_id, -1)]
    action.line_ex_set_bus = [(line_id, -1)]
    action.line_set_status = [(line_name, -1)]
    return action
