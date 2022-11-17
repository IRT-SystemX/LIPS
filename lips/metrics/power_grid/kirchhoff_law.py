# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union
import itertools
from cmath import exp, pi
import numpy as np

from ...logger import CustomLogger
#from lips.logger import CustomLogger

def verify_kirchhoff_law(predictions: dict,
                         log_path: Union[str, None]=None,
                         result_level: int=0,
                         **kwargs):
    """
    This function verifies the Kirchhoff's law based on power flow equations:
        
        - For active power
            ``$P_k = \sum_{j=1}^N |V_k||V_j|\left(G_{kj} cos(\theta_k - \theta_j) + B_{kj} sin(\theta_k - \theta_j)\right)$``

        - For reactive power
            ``$Q_k = \sum_{j=1}^N |V_k||V_j|\left(G_{kj} sin(\theta_k - \theta_j) - B_{kj} sin(\theta_k - \theta_j)\right)$``


    We compare the left hand side and right hand side of the power flow equations.
    The right hand side is computed based on predictions
    For the left hand side, power values from observations are used

    Parameters
    ----------
    predictions: ``dict``
        Predictions made by an augmented simulator
    log_path: ``str``, optional
        the path where the logs should be saved, by default None
    **kwargs: ``dict``
        It should contain `observations` and `config` to load the tolerance

    Returns
    -------
    ``dict``
        a dictionary with useful information about the verification

    These informations are:
    - mae_per_obs: `array`
        The Absolute error (difference) between two sides of equation per observation

    - mae: `scalar`
        the Mean Absolute Error (difference) between two sides of the equation

    - wmape: `scalar`
        weighted mean absolute percentage error between two sides of the equation

    - violation_proportion: `scalar`
        The percentage of violation of Joule law over all the observations on the basis of a tolerance indicated by the user

    Examples
    --------
    You can use this function like:

    .. code-block:: python

        import pathlib
        from lips.benchmark.powergridBenchmark import PowerGridBenchmark
        # indicate required paths
        LIPS_PATH = pathlib.Path().resolve()
        DATA_PATH = LIPS_PATH / "reference_data" / "powergrid" / "l2rpn_case14_sandbox"
        BENCH_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
        SIM_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "simulators"
        BASELINES_PATH = LIPS_PATH / "trained_baselines" / "powergrid"
        EVALUATION_PATH = LIPS_PATH / "evaluation_results" / "PowerGrid"
        LOG_PATH = LIPS_PATH / "lips_logs.log"

        # this function requires the third scenario or benchmark3
        benchmark3= PowerGridBenchmark(benchmark_path=None,
                                    benchmark_name="Benchmark3",
                                    load_data_set=False,
                                    config_path=BENCH_CONFIG_PATH,
                                    log_path=None)
        # generate some data
        benchmark3.generate(nb_sample_train=100,
                            nb_sample_val=1,
                            nb_sample_test=1,
                            nb_sample_test_ood_topo=1,
                            do_store_physics=True
                            )
        data = benchmark3.train_dataset.data
        env = benchmark3.training_simulator._simulator

        # For the verification purpose, real data are used as prediction and observation 
        # Feel free to use your predictions
        verifications = verify_kirchhoff_law(predictions=data, observations=data, env=env, tolerance=0.01)
    """
     # logger
    logger = CustomLogger("PhysicsCompliances(Joule_law)", log_path).logger
    
    try:
        env = kwargs["env"]
    except KeyError:
        logger.error("The requirements were not satisiftied to verify_joule_law function")
        raise

    try:
        config = kwargs["config"]
    except KeyError:
        try:
            tolerance = kwargs["tolerance"]
        except KeyError:
            logger.error("The tolerance could not be found for verify_joule_law function")
            # raise
            tolerance = 1e-1
            verify_reactive_power = False
        else:
            tolerance = float(tolerance)
    else:
        tolerance = float(config.get_option("eval_params")["KCL"]["tolerance"])
        verify_reactive_power = bool(config.get_option("eval_params")["KCL"]["verify_reactive_power"])

    try:
        observations = kwargs["observations"]
    except KeyError:
        logger.error("The requirements were not satisiftied to call verify_voltage_at_bus function")
        raise

    verifications = dict()
    ybuses = observations["YBus"]
    sbuses = observations["SBus"]
    prod_qs = observations["prod_q"]

    obs = env.reset()
    data_size = observations["prod_p"].shape[0]
    sbuses_after_pf = np.zeros_like(ybuses)
    bus_ps = np.zeros_like(ybuses)
    bus_qs = np.zeros_like(ybuses)
    for idx in range(data_size):
        obs.topo_vect = observations["topo_vect"][idx]
        ybus = ybuses[idx]
        sbus_init = sbuses[idx]

        #get SBus after powerflow: reactive power of generators have been computed and should be added
        sbus_after_pf = sbus_init
        prod_bus, _ = obs._get_bus_id(obs.gen_pos_topo_vect, obs.gen_to_subid)
        prod_q = prod_qs[idx]
        for i, prod_bus_item in enumerate(prod_bus):
            sbus_after_pf[prod_bus_item] += 1j * prod_q[i] / env.backend._grid.get_sn_mva()

        #bus_connectivity = get_bus_connectivity_full(obs)
        bus_connectivity = [list(np.where(row)[0]) for row in ybus]
        bus_v_complex, bus_theta = get_complex_v(env, obs, predictions, idx, ybus)
        bus_theta_rad = bus_theta * (pi / 180)
        bus_p = np.zeros(ybus.shape[0])
        bus_q = np.zeros(ybus.shape[0])
        for k, neighbor_buses in enumerate(bus_connectivity):
            tmp_p = 0
            tmp_q = 0
            for j in neighbor_buses:
                tmp_p += np.abs(bus_v_complex[k]) * np.abs(bus_v_complex[j]) * (ybus[k,j].real*np.cos(bus_theta_rad[k] - bus_theta_rad[j]) + ybus[k,j].imag*np.sin(bus_theta_rad[k] - bus_theta_rad[j]))
                tmp_q += np.abs(bus_v_complex[k]) * np.abs(bus_v_complex[j]) * (ybus[k,j].real*np.sin(bus_theta_rad[k] - bus_theta_rad[j]) - ybus[k,j].imag*np.cos(bus_theta_rad[k] - bus_theta_rad[j]))
            bus_p[k] = tmp_p
            bus_q[k] = tmp_q
        
        bus_ps[idx] = bus_p
        bus_qs[idx] = bus_q
        sbuses_after_pf[idx] = sbus_after_pf
        # the other more fast computation of 
        # SBus_computed = np.conj(np.matmul(ybus, bus_v_complex))*bus_v_complex
        # print("MAE between two S calculation:", np.mean(np.abs(bus_p + 1j*bus_q - SBus_computed)))
        # print(np.mean(np.abs(np.round(SBus_computed.real,2) - np.round(sbus_after_pf.real, 2))))
    
    # we check for all nodes execpt for the slack. For the slack, the active power has been adjusted to compensate for the losses, 
    # hence we should have adjusted for it in sbus_after_pf.
    # but we cannot recover the adjustment from the data we saved, so we ignore that index
    slack_id=prod_bus[-1]
    index_to_keep=np.ones(len(sbus_after_pf),dtype=bool)
    index_to_keep[slack_id]=False

    mae_active_per_obs = np.mean(np.abs(np.round(bus_ps,2) - np.round(sbuses_after_pf.real, 2))[:,index_to_keep], axis=1)
    mae_active = np.mean(mae_active_per_obs)
    wmape_active = np.mean(np.abs(np.round(bus_ps,2) - np.round(sbuses_after_pf.real, 2))[:,index_to_keep]) / np.mean(np.abs(np.round(sbuses_after_pf.real, 2))[:,index_to_keep])
    violation_prop_active = np.sum(mae_active_per_obs > tolerance) / mae_active_per_obs.size
    mae_reactive_per_obs = np.mean(np.abs(np.round(bus_qs,2) - np.round(sbuses_after_pf.imag, 2))[:,index_to_keep], axis=1)
    mae_reactive = np.mean(mae_reactive_per_obs)
    wmape_reactive = np.mean(np.abs(np.round(bus_qs,2) - np.round(sbuses_after_pf.imag, 2))[:,index_to_keep]) / np.mean(np.abs(np.round(sbuses_after_pf.imag, 2))[:,index_to_keep])
    violation_prop_reactive = np.sum(mae_reactive_per_obs > tolerance) / mae_reactive_per_obs.size

    logger.info("Kirchhoff law active power violation proportion: %.3f", violation_prop_active)
    logger.info("Kirchhoff law active power MAE: %.3f", mae_active)
    logger.info("Kirchhoff law active power WMAPE: %.3f", wmape_active)
    if verify_reactive_power:
        logger.info("Kirchhoff law reactive power violation proportion: %.3f", violation_prop_reactive)
        logger.info("Kirchhoff law reactive power MAE: %.3f", mae_reactive)
        logger.info("Kirchhoff law reactive power WMAPE: %.3f", wmape_reactive)

    if result_level > 0:
        verifications["mae_active_per_obs"] = np.round(mae_active_per_obs, 3)
        if verify_reactive_power:
            verifications["mae_reactive_per_obs"] = np.round(mae_reactive_per_obs, 3)
    verifications["violation_proportion_active"] = np.round(violation_prop_active,3)
    verifications["mae_active"] = np.round(mae_active, 3)
    verifications["wmape_active"] = np.round(wmape_active, 3)
    if verify_reactive_power:
        verifications["violation_proportion_reactive"] = np.round(violation_prop_reactive, 3)
        verifications["mae_reactive"] = np.round(mae_reactive, 3)
        verifications["wmape_reactive"] = np.round(wmape_reactive, 3)

    return verifications

def get_complex_v(env, obs, predictions, idx, ybus):
    lines_or_pu_to_kv=env.backend.lines_or_pu_to_kv
    lines_ex_pu_to_kv=env.backend.lines_ex_pu_to_kv
    v_or = predictions["v_or"][idx] / lines_or_pu_to_kv
    v_ex = predictions["v_ex"][idx] / lines_ex_pu_to_kv

    #we want to get V in complex form per bus_bar
    bus_theta = np.zeros(ybus.shape[0])
    bus_v=np.zeros(ybus.shape[0])

    lor_bus, _ = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
    lex_bus, _ = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)

    bus_theta[lor_bus] = predictions["theta_or"][idx]
    bus_theta[lex_bus] = predictions["theta_ex"][idx]

    bus_v[lor_bus]=v_or
    bus_v[lex_bus]=v_ex
    bus_v_complex=np.array([bus_v[k]*exp(1j*(bus_theta[k]*pi)/180) for k in range(len(bus_v))])
    return bus_v_complex, bus_theta

# Not used, To be removed
def get_bus_connectivity_full(obs) -> list:
    """
    This function returns the connectivity at bus level under an array of tuples
    
    Parameters
    ----------
    obs : ``grid2op.Observation``
        observation object of a grid2op environment

    Returns
    -------
    ``list``
        lists indicating the connectivity among buses
        e.g., a list [0,1,4] at index 0 of this array indicates that the bus 0 is connected to buses 1 and 4

    Examples
    --------

    You can use this function like:

    .. code-block:: python
        import grid2op
        env_name = ...

        env = grid2op.make(env_name)
        obs = env.reset()

        bus_connectivity = get_bus_connectivity_full(obs) # [[0, 1, 4], [1, 2, 3, 4, 0], [2, 3, 1], ... ]
    """
    connectivity_list = []
    lor_bus, _ = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
    lex_bus, _ = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)
    index_array = np.vstack((lor_bus, lex_bus)).T
    print(index_array)

    connectivity_list = list()
    for i in range(len(index_array)):
        tmp = list()
        tmp.append([i])
        tmp.append(list(index_array[index_array[:,0]==i][:,1]))
        tmp.append(list(index_array[index_array[:,1]==i][:,0]))
        tmp = list(itertools.chain(*tmp))
        connectivity_list.append(tmp)

    return connectivity_list

if __name__ == "__main__":
    import pathlib
    from lips.benchmark.powergridBenchmark import PowerGridBenchmark
    # indicate required paths
    LIPS_PATH = pathlib.Path().resolve()
    DATA_PATH = LIPS_PATH / "reference_data" / "powergrid" / "l2rpn_case14_sandbox"
    BENCH_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
    SIM_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "simulators"
    BASELINES_PATH = LIPS_PATH / "trained_baselines" / "powergrid"
    EVALUATION_PATH = LIPS_PATH / "evaluation_results" / "PowerGrid"
    LOG_PATH = LIPS_PATH / "lips_logs.log"

    benchmark3= PowerGridBenchmark(benchmark_path=None,
                                   benchmark_name="Benchmark3",
                                   load_data_set=False,
                                   config_path=BENCH_CONFIG_PATH,
                                   log_path=None)

    benchmark3.generate(nb_sample_train=100,
                        nb_sample_val=1,
                        nb_sample_test=1,
                        nb_sample_test_ood_topo=1,
                        do_store_physics=True
                        )

    data = benchmark3.train_dataset.data
    env = benchmark3.training_simulator._simulator

    verifications = verify_kirchhoff_law(predictions=data, observations=data, env=env)#, tolerance=0.01)
    print(verifications)
