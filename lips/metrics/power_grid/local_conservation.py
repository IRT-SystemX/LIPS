# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union
import numpy as np
from sklearn.metrics import mean_absolute_error

from ...logger import CustomLogger

def local_conservation(predictions,
                       log_path: Union[str, None]=None,
                       result_level: int=0,
                       **kwargs):
    """compute the conservation law for all the observations at each station

    Parameters
    ----------
    env : ``Grid2op.environment``
        Environment used for a specific benchmark
    observations : ``dict``
        Real observations dict
    predictions : ``dict``
        Predictions dict

    Returns
    -------
    _type_
        _description_
    """
    # logger
    logger = CustomLogger("PhysicsCompliances(LCE)", log_path).logger

    try:
        observations = kwargs["observations"]
    except KeyError:
        logger.error("The requirements were not satisiftied to verify_energy_conservation function")
        raise

    try:
        env = kwargs["env"]
    except KeyError:
        logger.error("The environment should be passed to this function, to be able to identify the connected elements.")
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
        tolerance = float(config.get_option("eval_params")["LC_tolerance"])

    try:
        prod_p = observations["prod_p"]
        load_p = observations["load_p"]
        p_or_real = observations["p_or"]
        p_ex_real = observations["p_ex"]
        p_or_pred = predictions["p_or"]
        p_ex_pred = predictions["p_ex"]

    except KeyError:
        logger.error("The observations or/and predictions do not include required variables")
        raise

    verifications = dict()

    data_size = len(prod_p)
    lc_array = np.zeros((data_size, env.n_sub), dtype=float)
    left_array = np.zeros((data_size, env.n_sub), dtype=float)
    right_array = np.zeros((data_size, env.n_sub), dtype=float)
    for id_ in range(data_size):
        lc_array[id_], left_array[id_], right_array[id_] = local_conservation_at_obs(env, prod_p, load_p, p_or_pred, p_ex_pred, obs_id=id_)

    failed_i, failed_j = np.where(abs(lc_array) > tolerance)
    failed_i = np.array(failed_i).reshape(-1, 1)
    failed_j = np.array(failed_j).reshape(-1, 1)
    violation_percentage = (len(failed_i) / lc_array.size)*100
    mae = np.mean(np.abs(left_array - right_array))
    wmape = np.mean(np.abs(left_array - right_array)) / np.mean(np.abs(left_array))

    logger.info("MAE for local conservation: %.3f", mae)
    logger.info("WMAPE for local conservation: %.3f", wmape)

    if result_level > 0:
        verifications["lc_values"] = lc_array
    verifications["violation_percentage"] = violation_percentage
    verifications["mae"] = mae
    verifications["mape"] = wmape

    return verifications


def local_conservation_at_obs(env, prod_p, load_p, p_or, p_ex, obs_id):
    """compute the law at observation level for all the substations

    Parameters
    ----------
    env : _type_
        _description_
    observations : _type_
        _description_
    predictions : _type_
        _description_
    obs_id : _type_
        _description_
    """
    lc_at_obs = np.zeros(env.n_sub, dtype=float)
    left_side = np.zeros(env.n_sub, dtype=float)
    right_side = np.zeros(env.n_sub, dtype=float)
    for sub_id in range(env.n_sub):
        lc_at_obs[sub_id], left_side[sub_id], right_side[sub_id] = local_conservation_at_obs_at_sub(env,
                                                                                          prod_p,
                                                                                          load_p,
                                                                                          p_or,
                                                                                          p_ex,
                                                                                          obs_id=obs_id,
                                                                                          sub_id=sub_id
                                                                                          )
    return lc_at_obs, left_side, right_side


def local_conservation_at_obs_at_sub(env, prod_p, load_p, p_or, p_ex, obs_id, sub_id):
    """verify the law at a specific observation and a substation

    Parameters
    ----------
    env : _type_
        _description_
    observations : _type_
        _description_
    predictions : _type_
        _description_
    obs_id : _type_
        _description_
    substation_id : _type_
        _description_
    """
    connectivity_dict = env.get_obj_connect_to(substation_id=sub_id)
    production_ = prod_p[obs_id, :][connectivity_dict["generators_id"]]
    load_ = load_p[obs_id, :][connectivity_dict["loads_id"]]
    p_or = p_or[obs_id, :][connectivity_dict["lines_or_id"]]
    p_ex = p_ex[obs_id, :][connectivity_dict["lines_ex_id"]]
    left_side = sum(production_) - sum(load_)
    right_side = sum(p_or) + sum(p_ex)
    lc_at_sub_at_obs =  left_side - right_side
    return lc_at_sub_at_obs, left_side, right_side
