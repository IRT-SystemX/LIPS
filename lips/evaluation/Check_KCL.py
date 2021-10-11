# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import numpy as np
import math


try:
    from lightsim2grid import PhysicalLawChecker
except ImportError as error:
    print("Install the LightSim2Grid to be able to use this verification")
    raise

def check_kcl(env, ref_obs, predictions, tol=1e-3):
    """
    This function used the check_solution function offered by lightsim2grid package, to 
    verify the Kirchhoff's current law

    params
    ------
        env: `grid2op` environment
            the grid2op environment used to generate the data

        ref_obs: `dict`
            the reference data used as ground truth

        predictions: `dict` 
            it should include the voltages (V) and the voltage angles (theta)

        tol: `float`
            the tolerance to respect the KCL

    returns
    _______
        KCL_value: `float`
            a global float value averaged over all the observations
        
        violation_prop_obs_level: `list`
            violation proportion at observation level

        violation_prop_node_level : `list`
            violation proportion at substation level
    """
    obs = env.reset()
    n_obs = len(predictions["v_or"])
    n_sub = env.n_sub

    # used to transform the Kv to Per Unit (Pu)
    lines_or_pu_to_kv = env.backend.lines_or_pu_to_kv
    lines_ex_pu_to_kv = env.backend.lines_ex_pu_to_kv

    # create a checker
    checker = PhysicalLawChecker(env)

    idx_list = list()
    nodes_violation_count = list()

    # loop over the observations
    for id_ in range(n_obs):
        # replacing the obs variables first by reference data
        for key_ in ref_obs.keys():
            setattr(obs, key_, ref_obs.get(key_)[id_, :])

        # replacing the obs variables finally by predictions
        for key_ in predictions.keys():
            setattr(obs, key_, predictions.get(key_)[id_, :])

        # the indices help to extract the V at substation level
        mat, (load_bus, gen_bus, stor_bus, lor_bus, lex_bus) = obs.flow_bus_matrix(active_flow=True, as_csr_matrix=False)

        #theta_or = predictions["theta_or"][id_]
        #theta_ex = predictions["theta_ex"][id_]
        #v_or = predictions["v_or"][id_] / lines_or_pu_to_kv
        #v_ex = predictions["v_ex"][id_] / lines_ex_pu_to_kv
        theta_or = getattr(obs, "theta_or")
        theta_ex = getattr(obs, "theta_ex")
        v_or = getattr(obs, "v_or") / lines_or_pu_to_kv
        v_ex = getattr(obs, "v_ex") / lines_ex_pu_to_kv

        v_list = list()
        theta_list = list()

        # computing the voltages and the angles at each substation (average)
        for sub_ in range(n_sub):
            tmp_v = np.mean(np.concatenate((v_or[lor_bus==sub_], v_ex[lex_bus==sub_])))
            v_list.append(tmp_v)
            tmp_theta = math.radians(np.mean(np.concatenate((theta_or[lor_bus==sub_], theta_ex[lex_bus==sub_]))))
            theta_list.append(tmp_theta)
        
        # computing the complex voltage at each sub station
        Va = np.array(theta_list) # voltage angles
        Vm = np.array(v_list) # voltage magnitudes
        # complex voltages
        v = np.zeros(2*n_sub, dtype=complex)
        V = Vm * np.exp(1j*Va)
        v[:n_sub] = V
        v[n_sub:] += 100 # TODO: maybe should be adapted when introducing topology changes
        # as it activates new bus bars. 

        mismatch = checker.check_solution(v, obs)
        
        # a case which does not respect the selected tolerance
        if np.any(mismatch > tol):
            idx_list.append(id_)
            nodes_violation_count.append(np.sum(mismatch > tol))

    # the proportion of observations including at least one violation at any node 
    violation_prop_obs_level = len(idx_list) / n_obs
    # the proportion of substations violating the KCL over all the observations
    violation_prop_node_level = sum(nodes_violation_count) / (n_obs*n_sub)

    return violation_prop_obs_level, violation_prop_node_level





        
        








