# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union
import numpy as np

from ...logger import CustomLogger

def global_conservation(predictions: dict,
                        log_path: Union[str, None]=None,
                        result_level: int=0,
                        **kwargs
                        ):
    """This function verifies the law of conservation of energy (LCE)

    productions = load + loss

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
    - LCE: `array`
        an array including the law of conservation of energy values stored for each step (observation)

    - violation_percentage: `scalar`
        a value expressed in percentage to indicate the percentage of

    - failed_indices: `array`
        an array giving the indices of observations that not verify the law given the indicated tolerance
    """
    # logger
    logger = CustomLogger("PhysicsCompliances(LCE)", log_path).logger

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
        tolerance = float(config.get_option("eval_params")["GC_tolerance"])


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

    left_array = productions - loads
    right_array = loss
    gce_ = left_array - right_array
    failed_indices = np.array(np.where(abs(gce_) > tolerance)).reshape(-1, 1)
    violation_percentage = (len(failed_indices) / len(gce_))*100
    mae = np.mean(np.abs(left_array - right_array))
    wmape = np.mean(np.abs(left_array - right_array)) / np.mean(np.abs(left_array))

    logger.info("MAE for global conservation: %.3f", mae)
    logger.info("WMAPE for global conservation: %.3f", mae)

    if result_level > 0:
        verifications["gc_values"] = gce_
    verifications["violation_percentage"] = violation_percentage
    verifications["mae"] = mae
    verifications["wmape"] = wmape

    return verifications
