# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import sys
from typing import Union
import math

import numpy as np
# TODO : implement a mean_absolute_error independently from sklearn
from sklearn.metrics import mean_absolute_error

from ...evaluation.utils import metric_factory
from ...metrics.power_grid.global_conservation import global_conservation
from ...metrics.power_grid.local_conservation import local_conservation
from ...metrics.power_grid.verify_voltage_equality import verify_voltage_at_bus
from ...metrics.power_grid.joule_law import verify_joule_law
from ...metrics.power_grid.ohm_law import verify_ohm_law
from ...metrics.power_grid.kirchhoff_law import verify_kirchhoff_law
from ...logger import CustomLogger

def verify_current_pos(predictions: dict,
                       log_path: Union[str, None]=None,
                       result_level: int=0,
                       **kwargs):
    """current positivity check

    Verifies the electrical current positivity at both extremity of power lines a_or >= 0 , a_ex >= 0

    Parameters
    ----------
    predictions: ``dict``
        dictionary of predictions made by an augmented simulator
    log_path: ``str``
        a path wehere the log should be saved
    **kwargs: ``dict``
        supplementary arguments (may be required in future)

    Returns
    -------
    `dict`
        a dictionary reporting the evaluation results for both line extremities
    """
    # logger
    logger = CustomLogger("PhysicsCompliances", log_path).logger
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
            if result_level > 0:
                verifications[key_]["indices"] = [(int(el[0]), int(el[1])) for el in a_or_errors]
            verifications[key_]["Error"] = float(error_a_or)
            verifications[key_]["Violation_proportion"] = float(a_or_violation_proportion)
        else:
            logger.info("Current positivity check passed for %s", key_)
    return verifications

def verify_voltage_pos(predictions:dict,
                       log_path: Union[str, None]=None,
                       result_level: int=0,
                       **kwargs):
    """Voltage positivity check

    Verifies the electrical voltage positivity at both extremity of power lines v_or >= 0 , v_ex >= 0

    Parameters
    ----------
    predictions: ``dict``
        dictionary of predictions made by an augmented simulator
    log_path: ``str``
        a path wehere the log should be saved
    **kwargs: ``dict``
        supplementary arguments (may be required in future)

    Returns
    -------
    ``dict``
        a dictionary reporting the evaluation results for both line extremities
    """
    # logger
    logger = CustomLogger("PhysicsCompliances", log_path).logger
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
            logger.info("the sum of negative voltage values (kV) for %s: %.3f", key_, error_v_or)
            if result_level > 0:
                verifications[key_]["indices"] = [(int(el[0]), int(el[1])) for el in v_or_errors]
            verifications[key_]["Error"] = float(error_v_or)
            verifications[key_]["Violation_proportion"] = float(v_or_violation_proportion)
        else:
            logger.info("Voltage positivity check passed for %s", key_)
    return verifications

def verify_loss_pos(predictions: dict,
                    log_path: Union[str, None]=None,
                    result_level: int=0,
                    **kwargs):
    """loss positivity check

    Verify that the electrical losses are greater than zero at each power line p_or + p_ex >= 0

    Parameters
    ----------
    predictions: ``dict``
        dictionary of predictions made by an augmented simulator
    log_path: ``str``
        a path wehere the log should be saved
    **kwargs: ``dict``
        supplementary arguments (may be required in future)

    Returns
    -------
    ``dict``
        a dictionary reporting the evaluation results
    """
    # logger
    logger = CustomLogger("PhysicsCompliances", log_path).logger
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
        if result_level > 0:
            verifications["loss_errors"] = [(int(el[0]), int(el[1])) for el in loss_errors]#float(loss_errors)
        verifications["violation_proportion"] = float(loss_violation_proportion)
    else:
        logger.info("Loss positivity check passed")
    return verifications

def verify_disc_lines(predictions: dict,
                      log_path: Union[str, None]=None,
                      result_level: int=0,
                      **kwargs):
    """Verifies if the predictions are null for disconnected lines

    Parameters
    ----------
    predictions: ``dict``
        dictionary of predictions made by an augmented simulator
    log_path: ``str``
        a path wehere the log should be saved
    **kwargs: ``dict``
        supplementary arguments

    Returns
    -------
    `dict`
        a dictionary reporting the evaluation results
    """
    FLOW_VARIABLES = ("p_or", "p_ex", "q_or", "q_ex", "a_or", "a_ex", "v_or", "v_ex")
    # logger
    logger = CustomLogger("PhysicsCompliances", log_path).logger
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
        num_var = 0
        for var_ in FLOW_VARIABLES:
            if var_ in predictions.keys():
                num_var += 1
                pred_disc = predictions[var_][ind_disc]
                violation = float(np.sum(np.abs(pred_disc)>0))
                verifications[var_] = violation / len_disc
                sum_disconnected_values += violation
        mean_disconnected_values = sum_disconnected_values / (len_disc * num_var)
        verifications["violation_proportion"] = mean_disconnected_values
    else:
        verifications["violation_proportion"] = 0.

    if sum_disconnected_values > 0:
        logger.info("Prediction in presence of line disconnection. Problem encountered !")
    else:
        logger.info("Prediction in presence of line disconnection. Check passed !")
    return verifications

def verify_current_eq(predictions: dict,
                      log_path: Union[str, None]=None,
                      result_level: int=0,
                      **kwargs):
    """
    verify the following relation between p, q and v :
    * a_or = sqrt(p_or**2 + q_or**2) / (sqrt(3).v_or)
    * a_ex = sqrt(p_ex**2 + q_ex**2) / (sqrt(3).v_ex)

    Todo
    ----
    TODO : update the equations by considering only voltage > 0 cases, hence it does not need eps

    Parameters
    ----------
    predictions: ``dict``
        dictionary of predictions made by an augmented simulator
    log_path: ``str``
        a path wehere the log should be saved
    **kwargs: ``dict``
        supplementary parameters (may be required in future)

    Returns
    -------
    ``dict``
        a dictionary reporting the evaluation results
    """
    # logger
    logger = CustomLogger("PhysicsCompliances", log_path).logger
    verifications = dict()
    # consider an epsilon value to avoid division by zero
    eps = sys.float_info.epsilon
    for key_ in ("_or", "_ex"):
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
        logger.info("Mean absolute error of a%s : %.3f", key_, np.mean(verifications["a"+key_+"_deviation"]))
    return verifications

def verify_loss(predictions,
                log_path: Union[str, None]=None,
                result_level: int=0,
                **kwargs):
    """Verify the energy loss

    The loss should be between 1 and 4 % of production at each step.

    2 possible way to call the function with two set of information:
    1) indicating only the path to the stored arrays by using the `path` parameter
    2) indicating explicitly the required variables for computing the law which are (prod_p, p_or and p_ex)

    Parameters
    ----------
    predictions: ``dict``
        Predictions made by an augmented simulator
    log_path: ``str``
        the path where the logs should be saved
    **kwargs: ``dict``
        It should contain `observations` and `config` to load the tolerance

    Returns
    -------
    ``dict``
        A dictionary comprising useful information about the loss

    The following keys are added to the dictionary:
    - EL: `array`
        array of energy losses for each iteration
    - violation_percentage: `float`
        percentage of violation of loss
    - failed_indices: `list`
        The indices of failed cases
    """
    # logger
    logger = CustomLogger("PhysicsCompliances(Loss)", log_path).logger
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


    verifications["violation_percentage"] = violation_percentage
    if result_level > 0:
        verifications["Law_values"] = el_
        verifications["failed_indices"] = failed_indices
    return verifications

metric_factory.register_metric("CURRENT_POS", verify_current_pos)
metric_factory.register_metric("VOLTAGE_POS", verify_voltage_pos)
metric_factory.register_metric("LOSS_POS", verify_loss_pos)
metric_factory.register_metric("CURRENT_EQ", verify_current_eq)
metric_factory.register_metric("DISC_LINES", verify_disc_lines)
metric_factory.register_metric("CHECK_LOSS", verify_loss)
metric_factory.register_metric("CHECK_GC", global_conservation)
metric_factory.register_metric("CHECK_LC", local_conservation)
metric_factory.register_metric("CHECK_VOLTAGE_EQ", verify_voltage_at_bus)
metric_factory.register_metric("CHECK_JOULE_LAW", verify_joule_law)
metric_factory.register_metric("CHECK_OHM_LAW", verify_ohm_law)
metric_factory.register_metric("CHECK_KCL", verify_kirchhoff_law)
