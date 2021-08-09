# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import numpy as np
from scipy.stats import pearsonr


def pearson_r(y_true, y_pred, multioutput="uniform", threshold=1.0):
    """
    Compute the pearson correlation coefficient given two matrices y_true and y_pred.

    Each row is supposed to be an example (on which the pearson correlation coefficient) is computed. The computation
    happens independently for each column.

    Notes
    ------
    If the difference 'max - min' for a column is bellow the threshold (by default 1.) this routines consider that
    the value is constant and thus do not compute the pearson correlation coefficient and returns "nan" instead.

    When multioutput is not "uniform" the average is computed only on the "non nan" variables. Nan component are
    ignored.

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
    rs: ``float`` or ``numpy.ndarray``
        If `multioutput` is "uniform" it will return a floating point number corresponding to the average pearson
        correlation coefficient, otherwise it will return a vector with as many component as the number of columns
        in y_true an y_pred.
    """
    if y_true.shape != y_pred.shape:
        raise RuntimeError("pearson_r can only be computed if y_true and y_pred have the same shape")
    if len(y_true.shape) != 2:
        raise RuntimeError("pearson_r can only be used with matrices")

    rs = np.zeros(y_true.shape[1])
    for col_id in range(y_true.shape[1]):
        true_tmp = y_true[:, col_id]
        pred_tmp = y_pred[:, col_id]

        # don't count some value considered "constant"
        is_ko_true = np.abs(np.max(true_tmp) - np.min(true_tmp)) <= threshold
        is_ko_pred = np.abs(np.max(pred_tmp) - np.min(pred_tmp)) <= threshold
        if is_ko_true or is_ko_pred:
            tmp_r = np.NaN
        else:
            tmp_r, *_ = pearsonr(true_tmp, pred_tmp)
        rs[col_id] = tmp_r

    if multioutput == "uniform":
        if np.all(~np.isfinite(rs)):
            # if everything is Nan then the returned value is Nan
            rs = np.NaN
        else:
            # don't take into account the nan there
            rs = np.nanmean(rs)
    return rs
