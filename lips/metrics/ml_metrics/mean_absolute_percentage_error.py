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

def mean_absolute_percentage_error(observations: dict, 
                                   predictions: dict, 
                                   output: str="average", 
                                   threshold=1., 
                                   log_path: Union[str, None]=None):
    """Mean Absolute Percentage Error

    Parameters
    ----------
    observations : ``dict``
        the true observations
    predictions : ``dict``
        the predictions of an augmented simulator
    output : ``str``, optional
        The multioutput type. It could be `raw` or `average`, by default "average"
    threshold : ``float``, optional
        the threshold to consider for the values, by default 1.
    log_path : Union[``str``, ``None``], optional
        the path to the log file, by default None

    Returns
    -------
    ``scalar`` or ``array``
        whether the average error or raw valued error per column

    Raises
    ------
    NotImplementedError
        other type of output is not implemented
    """
    # logger
    logger = CustomLogger("mean_absolute_error", log_path).logger

    if observations.shape != predictions.shape:
        logger.error("The observations and predictions shoud have the same shape for MAPE")
        raise RuntimeError("The observations and predictions shoud have the same shape for MAPE")
    
    epsilon = np.finfo(np.float64).eps
    error = np.abs(observations - predictions) / np.maximum(np.abs(observations), epsilon)
    if output == "raw":
        error = np.mean(error, axis=0)
    elif output == "average":
        error = np.mean(error)
    else:
        logger.error("the output mode is unknown. Use `raw` or `average`")
        raise ValueError("the output mode is unknown. Use `raw` or `average`")

    return error
