# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""
Function to compute the Ohm law for power grid use case
"""
from typing import Union
from cmath import exp, pi

import numpy as np

from ...logger import CustomLogger

def verify_ohm_law(predictions: dict,
                   log_path: Union[str, None]=None,
                   result_level: int=0,
                   **kwargs):
    """
    This function compute Ohm's law at the branch level

    We compute $S_{kj} = V_k \cdot Y_{kj}^* \cdot V_j^*$, where $S_{kj}$ designates the power
    conducted by the branch connecting node $k$ to $j$. As such a $N\times N$ matrix of node
    to node connection is created. Finally, we compare the ground truth matrix with predicted
    one.

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
        The Absolute error (difference) of Ohm law between ground truth and predictions computed per observation

    - mae_per_obj: `array`
        The Absolute Error (difference) between two sides of Ohm per branch for each observation

    - mae: `scalar`
        the Mean Absolute Error (difference) between predictions and ground truth

    - wmape: `scalar`
        weighted mean absolute percentage error between predictions and ground truth

    - violation_proportion: `scalar`
        The percentage of violation of Joule law over all the observations

    Examples
    --------
    You can use this function like:

    .. code-block:: python

        import pathlib
        from lips.metrics.power_grid.ohm_law import verify_ohm_law
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

        verifications = verify_ohm_law(predictions=data, observations=data, env=env, result_level=0)#, tolerance=0.01)
        print(verifications)
    """
    # logger
    logger = CustomLogger("PhysicsCompliances(Ohm_law)", log_path).logger

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
            logger.error("The tolerance could not be found for verify_ohm_law function. The default value will be used")
            # raise
            tolerance = 1e-1
        else:
            tolerance = float(tolerance)
    else:
        tolerance = float(config.get_option("eval_params")["OHM_tolerance"])

    try:
        observations = kwargs["observations"]
    except KeyError:
        logger.error("The requirements were not satisiftied to verify_ohm_law function")
        raise

    verifications = dict()
    ybuses =  observations["YBus"]
    p_ors = predictions["p_or"]
    p_exs = predictions["p_ex"]

    obs = env.reset()
    grid_model = env.backend._grid
    n_lines = len(grid_model.get_lines())
    data_size = ybuses.shape[0]
    p_or_computed = np.zeros_like(p_ors)
    p_ex_computed = np.zeros_like(p_exs)

    for idx in range(data_size):
        obs.topo_vect = observations["topo_vect"][idx]
        lor_bus, _ = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
        lex_bus, _ = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)
        ybus = ybuses[idx]
        bus_v_complex, _ = get_complex_v(env, obs, predictions, idx, ybus)

        for id_, line in enumerate(grid_model.get_lines()):
            # print("Line : ", id_)
            or_ = lor_bus[id_]
            ex_ = lex_bus[id_]
            # print("from {} to {}".format(or_, ex_))
            s_or = (ybus[or_,ex_]*bus_v_complex[ex_] - (ybus[or_,ex_] - 1j*line.h_pu*0.5)*bus_v_complex[or_]) * np.conj(bus_v_complex[or_])
            p_or = s_or.real * grid_model.get_sn_mva()
            p_or_computed[idx, id_] = p_or
            # print("p_or_computed: {} and predicted_p_or: {}".format(p_or, obs.p_or[id_]))
            # assert np.abs(p_or - obs.p_or[id_]) < 1e-4

            s_ex = (ybus[or_,ex_]*bus_v_complex[or_] - (ybus[or_,ex_] - 1j*line.h_pu*0.5)*bus_v_complex[ex_]) * np.conj(bus_v_complex[ex_])
            p_ex = s_ex.real * grid_model.get_sn_mva()
            p_ex_computed[idx, id_] = p_ex
            # print("p_ex_computed: {} and predicted_p_ex: {}".format(p_ex, obs.p_ex[id_]))
            # assert np.abs(p_ex - obs.p_ex[id_]) < 1e-4

        for id_trafo, trafo in enumerate(grid_model.get_trafos()):
            id_ = id_trafo + n_lines
            or_ = lor_bus[id_]
            ex_ = lex_bus[id_]

            s_or = (ybus[or_,ex_]*bus_v_complex[ex_] - (ybus[or_,ex_] - 1j*trafo.h_pu*0.5)*bus_v_complex[or_]) * np.conj(bus_v_complex[or_])
            p_or = s_or.real * grid_model.get_sn_mva()
            p_or_computed[idx, id_] = p_or

            s_ex = (ybus[or_,ex_]*bus_v_complex[or_] - (ybus[or_,ex_] - 1j*trafo.h_pu*0.5)*bus_v_complex[ex_]) * np.conj(bus_v_complex[ex_])
            p_ex = s_ex.real * grid_model.get_sn_mva()
            p_ex_computed[idx, id_] = p_ex

    mae_per_obj_p_or = np.abs(p_ors - p_or_computed)
    mae_per_line_p_or = np.mean(mae_per_obj_p_or, axis=0)
    mae_per_obs_p_or = np.mean(mae_per_obj_p_or, axis=1)
    mae_p_or = np.mean(mae_per_obj_p_or)
    wmape_p_or = np.mean(np.abs(p_ors - p_or_computed)) / np.mean(np.abs(p_ors))
    violation_prop_p_or = np.sum(mae_per_obj_p_or > tolerance) / mae_per_obj_p_or.size

    mae_per_obj_p_ex = np.abs(p_exs - p_ex_computed)
    mae_per_line_p_ex = np.mean(mae_per_obj_p_ex, axis=0)
    mae_per_obs_p_ex = np.mean(mae_per_obj_p_ex, axis=1)
    mae_p_ex = np.mean(mae_per_obj_p_ex)
    wmape_p_ex = np.mean(np.abs(p_exs - p_ex_computed)) / np.mean(np.abs(p_exs))
    violation_prop_p_ex = np.sum(mae_per_obj_p_ex > tolerance) / mae_per_obj_p_ex.size

    logger.info("Ohm's law violation proportion for p_or: %.3f", violation_prop_p_or)
    logger.info("Ohm's law violation proportion for p_ex: %.3f", violation_prop_p_ex)

    verifications["mae_p_or"] = mae_p_or
    verifications["mae_p_ex"] = mae_p_ex
    verifications["wmape_p_or"] = wmape_p_or
    verifications["wmape_p_ex"] = wmape_p_ex
    verifications["violation_prop_p_or"] = violation_prop_p_or
    verifications["violation_prop_p_ex"] = violation_prop_p_ex

    if result_level > 0:
        verifications["mae_per_line_p_or"] = mae_per_line_p_or
        verifications["mae_per_line_p_ex"] = mae_per_line_p_ex
    if result_level > 1:
        verifications["mae_per_obs_p_or"] = mae_per_obs_p_or
        verifications["mae_per_obs_p_ex"] = mae_per_obs_p_ex

    return verifications

def get_complex_v(env, obs, predictions, idx, ybus):
    lines_or_pu_to_kv = env.backend.lines_or_pu_to_kv
    lines_ex_pu_to_kv = env.backend.lines_ex_pu_to_kv
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

if __name__ == '__main__':
    import pathlib
    import copy
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
    # Disturb manually the predictions to verify the verification results
    # predictions = copy.deepcopy(data)
    # predictions["v_or"][1] = data["v_or"][0]
    # predictions["v_ex"][3] = data["v_ex"][10]
    env = benchmark3.training_simulator._simulator

    verifications = verify_ohm_law(predictions=data, observations=data, env=env, result_level=0)#, tolerance=0.01)
    print(verifications)
