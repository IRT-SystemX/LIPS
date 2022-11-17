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

def normalized_mean_squared_error(observations: dict,
                                  predictions: dict,
                                  output: str="average",
                                  squared: bool=False,
                                  normalization: str="maxmin",
                                  log_path: Union[str, None]=None):
    """Normalized Mean Squared Error

    Parameters
    ----------
    observations : ``dict``
        the true observations
    predictions : ``dict``
        the predictions of an augmented simulator
    output : ``str``, optional
        The multioutput type. It could be `raw` or `average`, by default "average"
    squared : ``bool``, optional
        Whether to square or not. If ``False`` Root Mean Squared Error is computed, by default ``False``
    normalization : ``str``, optional
        Normalization type. Possible normalizations are: `sd`, `maxmin`, `mean`, `iq`, by default "maxmin"
    log_path : Union[``str``, ``None``], optional
        the path to the log file, by default ``None``

    Returns
    -------
    ``scalar`` or ``array``
        whether the average error or raw valued error per column

    Raises
    ------
    NotImplementedError
        other type of normalization not implemented
    """
    # logger
    logger = CustomLogger("mean_squared_error", log_path).logger
    
    if observations.shape != predictions.shape:
        logger.error("The observations and predictions shoud have the same shape for NRMSE")
        raise RuntimeError("The observations and predictions shoud have the same shape for NRMSE")
    
    epsilon = 1.#np.finfo(np.float64).eps
    
    error = (observations - predictions)**2
    error = np.mean(error, axis=0)

    if normalization == "sd":
        normalization_factor = np.maximum(np.std(observations, axis=0), epsilon)
    elif normalization == "mean":
        normalization_factor = np.mean(observations, axis=0)
    elif normalization == "maxmin":
        normalization_factor = np.maximum(np.max(observations, axis=0)-np.min(observations, axis=0), epsilon)
    elif normalization == "iq":
        normalization_factor = np.maximum(np.quantile(observations, q=0.75, axis=0)-np.quantile(observations, q=0.25, axis=0), epsilon)
    else:
        logger.error("this type of normalization factor is not implemented.")
        raise NotImplementedError("this type of normalization factor is not implemented.")

    if not(squared):
        error = np.sqrt(error)

    error /= normalization_factor

    if output == "average":
        error = np.mean(error)

    return error
