# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import numpy as np
import warnings


def mape(y_true, y_pred, multioutput="uniform", threshold=1.):
    """
    Computes the "MAPE" norm (mean absolute percentage error).

    if multioutput is "uniform" then a single number is returned and it consists in taking the average of the previous
    vector.

    Notes
    ------
    This function completely ignores the values where `y_true` are such that `|y_true| < threshold`. It only considers
    values of y_true above the `threshold`.

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
    mape_: ``float`` or ``numpy.ndarray``
        If `multioutput` is "uniform" it will return a floating point number, otherwise it will return a vector
        with as many component as the number of columns in y_true an y_pred.

    """
    if y_true.shape != y_pred.shape:
        raise RuntimeError("mape can only be computed if y_true and y_pred have the same shape")

    try:
        threshold = float(threshold)
    except Exception as exc_:
        raise exc_
    if threshold < 0.:
        raise RuntimeError("The threshold should be a positive floating point value.")

    index_ok = (np.abs(y_true) > threshold)
    rel_error_ = np.full(y_true.shape, fill_value=np.NaN, dtype=float)
    rel_error_[index_ok] = (y_pred[index_ok] - y_true[index_ok])/y_true[index_ok]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mape_ = np.nanmean(np.abs(rel_error_), axis=0)
    if multioutput == "uniform":
        mape_ = np.mean(mape_)
    return mape_
