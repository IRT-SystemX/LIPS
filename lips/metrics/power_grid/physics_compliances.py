# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import sys
from typing import Union
from collections import Counter
import math

import numpy as np
# TODO : implement a mean_absolute_error independently from sklearn
from sklearn.metrics import mean_absolute_error

from lips.logger import CustomLogger

def verify_current_pos(predictions: dict,
                       log_path: Union[str, None]=None,
                       **kwargs):
    """
    Verifies the electrical current positivity at both extremity of power lines a_or >= 0 , a_ex >= 0

    Parameters
    ==========
    predictions: `dict`
        dictionary of predictions made by an augmented simulator
    log_path: `str`
        a path wehere the log should be saved
    **kwargs:
        supplementary arguments (may be required in future)

    Returns
    =======
    verifications: `dict`
        a dictionary reporting the evaluation results for both line extremities
    """
    # logger
    logger = CustomLogger("Verify Current Positivity", log_path).logger
    verifications = dict()

    for key_ in ["a_or", "a_ex"]:
        try:
            a_arr = predictions[key_]
        except KeyError:
            logger.error("%s does not exists in predictions dict", key_)
            raise
        if np.any(a_arr < 0):
            verifications[key_] = {}
            a_or_errors = np.array(np.where(a_arr < 0)).T
            a_or_violation_proportion = (1.0 * len(a_or_errors)) / a_arr.size
            error_a_or = -np.sum(np.minimum(a_arr.flatten(), 0.))
            logger.info("the sum of negative current values (A) for %s: %.3f", key_, error_a_or)
            verifications[key_]["indices"] = [(int(el[0]), int(el[1])) for el in a_or_errors]
            verifications[key_]["Error"] = float(error_a_or)
            verifications[key_]["Violation_proportion"] = float(a_or_violation_proportion)
        else:
            logger.info("Current positivity check passed for %s", key_)
    return verifications

def verify_voltage_pos(predictions:dict,
                       log_path: Union[str, None]=None,
                       **kwargs):
    """
    Verifies the electrical voltage positivity at both extremity of power lines v_or >= 0 , v_ex >= 0

    Parameters
    ==========
    predictions: `dict`
        dictionary of predictions made by an augmented simulator
    log_path: `str`
        a path wehere the log should be saved
    **kwargs:
        supplementary arguments (may be required in future)

    Returns
    =======
    verifications: `dict`
        a dictionary reporting the evaluation results for both line extremities
    """
    # logger
    logger = CustomLogger("verify_voltage_pos", log_path).logger
    verifications = dict()

    for key_ in ["v_or", "v_ex"]:
        try:
            v_arr = predictions[key_]
        except KeyError:
            logger.error("%s does not exists in predictions dict", key_)
            raise
        if np.any(v_arr < 0):
            verifications[key_] = {}
            v_or_errors = np.array(np.where(v_arr < 0)).T
            v_or_violation_proportion = len(v_or_errors) / v_arr.size
            error_v_or = -np.sum(np.minimum(v_arr.flatten(), 0.))
            logger.info("the sum of negative voltage values (kV) at origin: %.3f", error_v_or)
            verifications[key_]["indices"] = [(int(el[0]), int(el[1])) for el in v_or_errors]
            verifications[key_]["Error"] = float(error_v_or)
            verifications[key_]["Violation_proportion"] = float(v_or_violation_proportion)
        else:
            logger.info("Voltage positivity check passed for %s", key_)
    return verifications

def verify_loss_pos(predictions: dict,
                    log_path: Union[str, None]=None,
                    **kwargs):
    """
    Verify that the electrical losses are greater than zero at each power line p_or + p_ex >= 0

    Parameters
    ==========
    predictions: `dict`
        dictionary of predictions made by an augmented simulator
    log_path: `str`
        a path wehere the log should be saved
    **kwargs:
        supplementary arguments (may be required in future)

    Returns
    =======
    verifications: `dict`
        a dictionary reporting the evaluation results
    """
    # logger
    logger = CustomLogger("verify_loss_pos", log_path).logger
    verifications = dict()
    try:
        p_or = predictions["p_or"]
        p_ex = predictions["p_ex"]
    except KeyError:
        logger.error("The predictions dict does not include required variables")
        raise
    loss = p_or + p_ex

    if np.any(loss):
        loss_error = -np.sum(np.minimum(loss, 0.))
        loss_errors = np.array(np.where(loss < 0)).T
        loss_violation_proportion = len(loss_errors) / p_or.size
        logger.info("the sum of negative losses : %.3f", loss_error)
        verifications["loss_criterion"] = loss_error#[(int(el[0]), int(el[1])) for el in loss_error]
        verifications["loss_errors"] = [(int(el[0]), int(el[1])) for el in loss_errors]#float(loss_errors)
        verifications["violation_proportion"] = float(loss_violation_proportion)
    else:
        logger.info("Loss positivity check passed")
    return verifications

def verify_disc_lines(predictions: dict,
                      log_path: Union[str, None]=None,
                      **kwargs):
    """
    Verifies if the predictions are null for disconnected lines

    Parameters
    ==========
    predictions: `dict`
        dictionary of predictions made by an augmented simulator
    log_path: `str`
        a path wehere the log should be saved
    **kwargs:
        supplementary arguments

    Returns
    =======
    verifications: `dict`
        a dictionary reporting the evaluation results
    """
    FLOW_VARIABLES = ("p_or", "p_ex", "q_or", "q_ex", "a_or", "a_ex")
    # logger
    logger = CustomLogger("verify_disc_lines", log_path).logger
    try:
        observations = kwargs["observations"]
    except KeyError:
        logger.error("Kwargs of function verify_disc_lines does not include observations key")

    try:
        line_status = observations["line_status"]
    except KeyError:
        logger.error("line_status key not found in observations.")

    verifications = dict()
    sum_disconnected_values = 0

    ind_disc = line_status != 1
    len_disc = np.sum(ind_disc)
    
    if np.any(ind_disc):
        for g, key_pairs in zip(("p", "q", "a"), FLOW_VARIABLES):
            if key_pairs[0] in predictions.keys():
                for key_ in key_pairs:
                    pred_disc = predictions[key_][ind_disc]
                    violation = float(np.sum(np.abs(pred_disc)>0))
                    verifications[key_] = violation
                    sum_disconnected_values += violation
                verifications[g] = float(np.sum((np.abs(key_pairs[0]) +
                                                        np.abs(key_pairs[1]))>0) /
                                                        len_disc)

    if sum_disconnected_values > 0:
        logger.info("Prediction in presence of line disconnection. Problem encountered !")
    else:
        logger.info("Prediction in presence of line disconnection. Check passed !")
    return verifications

def verify_current_eq(predictions: dict, 
                      log_path: Union[str, None]=None,
                      **kwargs):
    """
    verify the following relation between p, q and v : 
        * a_or = sqrt(p_or**2 + q_or**2) / (sqrt(3).v_or) 
        * a_ex = sqrt(p_ex**2 + q_ex**2) / (sqrt(3).v_ex)
    
    # TODO : update the equations by considering only voltage > 0 cases, hence it does not need eps

    Parameters
    ==========
    predictions: `dict`
        dictionary of predictions made by an augmented simulator
    log_path: `str`
        a path wehere the log should be saved
    **kwargs:
        supplementary parameters (may be required in future)

    Returns
    =======
    verifications: `dict`
        a dictionary reporting the evaluation results
    """
    # logger
    logger = CustomLogger("verify_current_eq", log_path).logger
    verifications = dict()
    # consider an epsilon value to avoid division by zero
    eps = sys.float_info.epsilon
    for key_ in ("or", "ex"):
        try:
            a_arr = predictions["a"+key_]
            p_arr = predictions["p"+key_]
            q_arr = predictions["q"+key_]
            v_arr = predictions["v"+key_]
        except KeyError:
            logger.error("The predictions dict does not include required variables")
            raise
        #a_or = sqrt(p_or**2 + q_or**2) / (sqrt(3).v_or)
        #a_ex = sqrt(p_ex**2 + q_ex**2) / (sqrt(3).v_ex)
        a_comp = (np.sqrt(p_arr**2 + q_arr**2) / ((np.sqrt(3) * v_arr)+eps)) * 1000
        verifications["a"+key_+"_deviation"] = [float(el) for el in
            mean_absolute_error(a_arr, a_comp, multioutput='raw_values')]
    return verifications

def verify_loss(predictions,
                log_path: Union[str, None]=None,
                **kwargs):
    """
    Verifies the energy loss. The loss should be between 1 and 4 % of production at each step.

    2 possible way to call the function with two set of information:
        1) indicating only the path to the stored arrays by using the `path` parameter
        2) indicating explicitly the required variables for computing the law which are (prod_p, p_or and p_ex)

    Parameters
    ==========
    predictions: `dict`
        Predictions made by an augmented simulator
    log_path: `str`
        the path where the logs should be saved
    **kwargs: 
        It should contain `observations` and `config` to load the tolerance

    Returns
    =======
    verifications: `dict` 
        It include following keys:
    
        EL: `array`
            array of energy losses for each iteration
        violation_percentage: `float`
            percentage of violation of loss
        failed_indices: `list`
            The indices of failed cases
    """
    # logger
    logger = CustomLogger("Verify Loss Eq", log_path).logger
    try:
        observations = kwargs["observations"]
    except KeyError:
        logger.error("The requirements were not satisiftied to call verify_loss function")
        raise
    try:
        config = kwargs["config"]
    except KeyError:
        try:
            tolerance = kwargs["tolerance"]
        except KeyError:
            logger.error("The tolerance could not be found for verify_loss function")
            raise
        else:
            tolerance = float(tolerance)
    else:
        tolerance = float(config.get_option("eval_params")["EL_tolerance"])
    

    verifications = dict()
    failed_indices = None
    violation_percentage = None
    

    try:
        prod_p = observations["prod_p"]
        p_or = predictions["p_or"]
        p_ex = predictions["p_ex"]
    except KeyError:
        logger.error("The observations or/and predictions do not include required variables")
        raise

    el_ = np.abs(np.sum(p_or + p_ex, axis=1))
    ratio = el_ / np.sum(prod_p, axis=1)

    condition = ratio > tolerance
    if np.any(condition):
        failed_indices = np.array(np.where(condition)).reshape(-1, 1)
        violation_percentage = (len(failed_indices) / len(el_))*100
        logger.info("Number of failed cases is: %i and the proportion is: %.3f %%", len(failed_indices), violation_percentage)
    else:
        logger.info("Verification is done without any violation")
        violation_percentage = 0.
    
    verifications["Law_values"] = el_
    verifications["violation_percentage"] = violation_percentage
    verifications["failed_indices"] = failed_indices
    return verifications

def verify_energy_conservation(predictions: dict,
                               log_path: Union[str, None]=None,
                               **kwargs
                              ):
    """
    This function verifies the law of conservation of energy (LCE) that says : productions = load + loss

    Parameters
    ==========
    predictions: `dict`
        Predictions made by an augmented simulator
    log_path: `str`
        the path where the logs should be saved
    **kwargs:
        It should contain `observations` and `config` to load the tolerance

    Returns
    =======
    verifications: `dict`
        a dictionary with following keys: 
        - LCE: `array`
            an array including the law of conservation of energy values stored for each step (observation)

        - violation_percentage: `scalar`
            a value expressed in percentage to indicate the percentage of 

        - failed_indices: `array`
            an array giving the indices of observations that not verify the law given the indicated tolerance

    """
    # logger
    logger = CustomLogger("Energy Conservation", log_path).logger

    try:
        observations = kwargs["observations"]
    except KeyError:
        logger.error("The requirements were not satisiftied to verify_energy_conservation function")
        raise

    try:
        config = kwargs["config"]
    except KeyError:
        try:
            tolerance = kwargs["tolerance"]
        except KeyError:
            logger.error("The tolerance could not be found for verify_loss function")
            raise
        else:
            tolerance = float(tolerance)
    else:
        tolerance = float(config.get_option("eval_params")["EL_tolerance"])
    

    try:
        prod_p = observations["prod_p"]
        load_p = observations["load_p"]
        p_or = predictions["p_or"]
        p_ex = predictions["p_ex"]
    except KeyError:
        logger.error("The observations or/and predictions do not include required variables")
        raise

    verifications = dict()
    productions = np.sum(prod_p, axis=1)
    loads = np.sum(load_p, axis=1)
    loss = np.sum(p_or + p_ex, axis=1) # TODO : not np.abs(np.sum(...)) ? Try and see

    lce_ = productions - loads - loss
    failed_indices = np.array(np.where(abs(lce_) > tolerance)).reshape(-1, 1)
    violation_percentage = (len(failed_indices) / len(lce_))*100
    criteria = mean_absolute_error(loads - productions, loss)

    logger.info("Mean Absolute Error (MAE) between (loads - productions) and loss is : %.3f", criteria)
    verifications["Law_values"] = lce_
    verifications["violation_percentage"] = violation_percentage
    verifications["criteria"] = criteria

    return verifications

def verify_kcl(env, ref_obs, predictions, tol=1e-3):
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
    try:
        from lightsim2grid import PhysicalLawChecker
    except ImportError as error:
        print("Install the LightSim2Grid to be able to use this verification")
        raise error

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
    # a set of attributes available in ref_obs but not in predictions (injections + topology)
    # it concerns the attributs which are used as predictors
    ref_obs_keys = ref_obs.keys() - predictions.keys()
    # loop over the observations
    for id_ in range(n_obs):
        # replacing the `obs` attributes using the injections + topology (real data)
        for key_ in ref_obs_keys:
            if key_ == "prod_p":
                key_1 = "gen_p"
            elif key_ == "prod_q":
                key_1 = "gen_q"
            elif key_ == "prod_v":
                key_1 = "gen_v"
            else:
                key_1 = key_
            setattr(obs, key_1, ref_obs.get(key_)[id_, :])

        # replacing the remaining attributes using the predictions
        for key_ in predictions.keys():
            if key_ == "prod_p":
                key_1 = "gen_p"
            elif key_ == "prod_q":
                key_1 = "gen_q"
            elif key_ == "prod_v":
                key_1 = "gen_v"
            else:
                key_1 = key_
            setattr(obs, key_1, predictions.get(key_)[id_, :])

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