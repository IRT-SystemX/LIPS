# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import numpy as np
import grid2op
import logging

logging.basicConfig(filename="logs.log",
                    level=logging.INFO,
                    format="%(levelname)s:%(message)s")


def Check_Kirchhoff_current_law(env=None,
                                env_name='l2rpn_case14_sandbox',
                                data=None,
                                load_p=None,
                                load_q=None,
                                prod_p=None,
                                prod_v=None,
                                line_status=None,
                                topo_vect=None,
                                active_flow=True,
                                tolerance=1e-2):
    """
    Verify the Kirchhoff's current law for the predicted observations
    It can be integrated as a function in Benchmark class, in this case,
    it does not required to give a path for predictions and it use the 
    values on the fly after prediction step.

    In the current version, we need a path to stored variables (input and outputs and topology)

    params
    ------
        env : Grid2op `environment`
            the environment used to generate the data

        env_name : `string`
            indicating the name of experimented environment. To be indicated if env is None

        data: `dict`
            a dictionary comprising the variables that are used as the output of a model

        prod_p : `list` of `array`
            a list of production active powers for different steps. Each item in the list
            corresponds to a unique sample and each sample has a length of env.n_gen.

        prod_v : `list` of `array`
            a list of production voltages for different steps. Each item in the list
            corresponds to a unique sample and each sample has a length of env.n_gen

        load_p : `list` of `array`
            a list of load active powers for different steps. Each item in the list
            corresponds to a unique sample and each sample has a length of env.n_load

        load_q : `list` of `array`
            a list of load reactive powers for different steps. Each item in the list
            corresponds to a unique sample and each sample has a length of env.n_load

        line_status : `list` of `array`
            a list of line statuses for different steps. Each item in the list
            corresponds to a unique sample and each sample has a length of n_lines

        topo_vect : `list` of `array`
            a list of topology vectors for different steps. Each item in the list
            corresponds to a unique sample and each sample has a length of (n_gen + n_load + n_lines * 2)

        active_flow : `boolean`
            if True verify the kirchhoff's law for active powers p, and if False verify the 
            current law for reactive power q

        tolerance : `float`
            the current law greater than tolerance will be considered as not verified


    Returns
    -------
        current_law_values_at_nodes: `list`
            list of arrays containing the current_law values at the bus level and each step

        current_law_values_network: `list`
            list of scalars containing sum of current_law values over all the nodes at each step

        current_law_not_verified: `list`
            list of observation indices which does not verify the Kirchhoff's current law
            Its length corresponds to the number of cases that do not respect the law

    """
    #print("************* Check kirchhoff's current law *************")
    a_or = data["a_or"]
    a_ex = data["a_ex"]
    q_or = data["q_or"]
    q_ex = data["q_ex"]
    p_or = data["p_or"]
    p_ex = data["p_ex"]
    v_or = data["v_or"]
    v_ex = data["v_ex"]
    load_v = data["load_v"]
    prod_q = data["prod_q"]
    prod_p_init = 1.0 * prod_p
    if "__prod_p_dc" in data:
        # specific case of the dc approximation that neglects the loss
        prod_p = data["__prod_p_dc"]

    len_obs = len(a_or)
    current_law_values_at_nodes = list()
    current_law_values_network = list()
    current_law_not_verified = list()
    violation_counter = 0  # number of violations of the law
    total_buses = 0  # number of buses, it can be changed wrt to topology

    if env is None:
        env = grid2op.make(env_name)
    obs = env.reset()

    for ind in range(len_obs):
        # fill the variables with real observations or predictions
        obs.a_or = a_or[ind]
        obs.a_ex = a_ex[ind]
        obs.q_or = q_or[ind]
        obs.q_ex = q_ex[ind]
        obs.p_or = p_or[ind]
        obs.p_ex = p_ex[ind]
        obs.v_or = v_or[ind]
        obs.v_ex = v_ex[ind]
        obs.load_v = load_v[ind]
        obs.gen_q = prod_q[ind]
        # inputs
        obs.load_p = load_p[ind]
        obs.load_q = load_q[ind]
        obs.gen_p = prod_p[ind]
        obs.gen_v = prod_v[ind]
        obs.line_status = np.asarray(line_status[ind], dtype=bool)
        obs.topo_vect = np.asarray(topo_vect[ind], dtype=int)

        # Compute the flow matrix
        flow_mat, _ = obs.flow_bus_matrix(active_flow=active_flow, as_csr_matrix=False)


        # the sum of rows gives an indication of how the Kirchhof's current law is respected
        tmp = flow_mat.sum(axis=1)
        # store the values at the node level
        current_law_values_at_nodes.append(tmp)
        # store the values over all the nodes
        current_law_values_network.append(sum(abs(tmp)))
        # verify if the current law is verified wrt considered tolerance
        # if sum(abs(tmp)) > tolerance:
        if np.sum(abs(tmp) > tolerance) > 0:
            current_law_not_verified.append(ind)
        violation_counter += np.sum(abs(tmp) > tolerance)
        # number of nodes depends on the topology, why I have considered the total_buses variable
        total_buses += len(tmp)

    violation_percentage = (violation_counter/total_buses)*100
    #print("{:.2f}% of nodes diverge from the Kirchhoff's current law with a magnitude more than {}MW tolerance".format(
    #    violation_percentage, tolerance))
    logging.info("{:.2f}% of nodes diverge from the Kirchhoff's current law with a magnitude more than {}MW tolerance".format(
        violation_percentage, tolerance))
    #print("{:.2f}% of {} not verify the Kirchhoff's current law at {} tolerance".format((len(current_law_not_verified) / len_obs) * 100, choice, tolerance))

    return current_law_values_at_nodes, current_law_values_network, current_law_not_verified, violation_percentage
