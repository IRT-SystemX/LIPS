# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import numpy as np


def nrmse(y_true, y_pred, multioutput="uniform", threshold=1.):
    """
    Computes the "normalized RMSE" norm.

    This norms compute the mean squared error (average over the rows of the square of the difference between y_true
    and y_pred). This gives a vector with as many elements as the number of columns of y_true and y_pred.

    Then it computes the RMSE by taking the square root of each numbers.

    Then, this rmse vector is divided by the the 'max - min' over each column, which is the "normalization".

    if multioutput is "uniform" then a single number is returned and it consists in taking the average of the previous
    vector.

    Notes
    ------
    If the difference 'max - min' for a column is bellow the threshold (by default 1.) it is replaced by this threshold.
    This has been implemented to avoid numerical instability when some columns are almost equal everywhere (this can
    happen in the power system settings: for example some voltages are always constant). You can put "0." to
    deactivate this feature.

    The shapes of y_true and y_pred should match.

    Parameters
    ----------
    y_true: ``numpy.ndarray``
        The true values. Each rows is an example, each column is a variable.

    y_pred: ``numpy.ndarray``
        The predicted values. Its shape should match the one from `y_true`

    multioutput: ``str``
        Whether or not to aggregate the returned values

    threshold: ``float``
        The the section "Notes" for more information. This should be a floating point number >= 0.

    Returns
    -------
    nrmse_: ``float`` or ``numpy.ndarray``
        If `multioutput` is "uniform" it will return a floating point number, otherwise it will return a vector
        with as many component as the number of columns in y_true an y_pred.

    """
    if y_true.shape != y_pred.shape:
        raise RuntimeError("nrmse can only be computed if y_true and y_pred have the same shape")

    try:
        threshold = float(threshold)
    except Exception as exc_:
        raise exc_
    if threshold < 0.:
        raise RuntimeError("The threshold should be a positive floating point value.")

    se_ = (y_true - y_pred)**2
    mse = np.mean(se_, axis=0)
    rmse = np.sqrt(mse)
    norm_ = (np.max(y_true, axis=0) - np.min(y_true, axis=0))
    norm_[norm_ <= threshold] = threshold
    nrmse_ = rmse / norm_
    if multioutput == "uniform":
        nrmse_ = np.mean(nrmse_)
    return nrmse_
