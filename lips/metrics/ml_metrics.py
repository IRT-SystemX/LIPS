# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
"""
These functions are from :
    https://github.com/BDonnot/leap_net/blob/master/leap_net/metrics/mape.py
    https://github.com/BDonnot/leap_net/blob/master/leap_net/metrics/mape_quantile.py
    https://github.com/BDonnot/leap_net/blob/master/leap_net/metrics/nrmse.py
"""

from typing import Union
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape(y_true, y_pred, multioutput="uniform", threshold=1.) -> Union[float, np.ndarray]:
    """
    Computes the "MAPE" norm (mean absolute percentage error).
    if multioutput is "uniform" then a single number is returned and it consists in taking the average of the previous
    vector.

    Notes
    -----
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
    Union[float, np.ndarray]
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

def mape_quantile(y_true, y_pred, multioutput="uniform", quantile=0.1) -> Union[float, np.ndarray]:
    """
    Computes the "MAPE" norm (mean absolute percentage error) but only on the `q` highest values column wise.
    This is a domain specific metric, used for example when we are interested in predicting correctly
    the highest values of a given variable.
    if multioutput is "uniform" then a single number is returned and it consists in taking the average of the previous
    vector.

    Notes
    -----
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
    quantile: ``float``
        The highest ratio to keep. For example, if `quantile=0.1` (default) the 10% highest values are kept.

    Returns
    -------
    Union[float, np.ndarray]
        If `multioutput` is "uniform" it will return a floating point number, otherwise it will return a vector
        with as many component as the number of columns in y_true an y_pred.

    """
    if y_true.shape != y_pred.shape:
        raise RuntimeError("mape can only be computed if y_true and y_pred have the same shape")

    try:
        threshold = float(quantile)
    except Exception as exc_:
        raise exc_
    if threshold < 0.:
        raise RuntimeError("The threshold should be a positive floating point value.")

    if threshold <= 0.:
        raise RuntimeError(f"The quantile `q` should be > 0: {quantile} found")
    if threshold >= 1.:
        raise RuntimeError(f"The quantile `q` should be < 1: {quantile} found")

    index_ok = (np.abs(y_true) > threshold)
    rel_error_ = np.full(y_true.shape, fill_value=np.NaN, dtype=float)
    rel_error_[index_ok] = (y_pred[index_ok] - y_true[index_ok])/y_true[index_ok]
    # what we want to do, but does not always keep the right number of rows
    # (especially when there are equalities...)
    # quantile_ytrue = np.percentile(y_true, q=100.*(1. - quantile), axis=0)
    # rel_error_quantile = rel_error_[(y_true > quantile_ytrue).reshape(y_true.shape)].reshape((-1, y_true.shape[1]))
    # compute how many values to keep
    nb_el_to_keep = int(quantile*y_pred.shape[0])
    nb_el_to_keep = max(nb_el_to_keep, 1)
    # keep the k ith values
    index_highest = np.argpartition(np.abs(y_true), axis=0, kth=-nb_el_to_keep)[-nb_el_to_keep:]
    rel_error_quantile = rel_error_[index_highest, np.arange(y_true.shape[1]).T]
    # compute the mape on these errors
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mape_quantile = np.nanmean(np.abs(rel_error_quantile), axis=0)
    if multioutput == "uniform":
        mape_quantile = np.mean(mape_quantile)
    return mape_quantile

def nrmse(y_true, y_pred, multioutput="uniform", threshold=1.) -> Union[float, np.ndarray]:
    """
    Computes the "normalized RMSE" norm.
    This norms compute the mean squared error (average over the rows of the square of the difference between y_true
    and y_pred). This gives a vector with as many elements as the number of columns of y_true and y_pred.
    Then it computes the RMSE by taking the square root of each numbers.
    Then, this rmse vector is divided by the the 'max - min' over each column, which is the "normalization".
    if multioutput is "uniform" then a single number is returned and it consists in taking the average of the previous
    vector.

    Notes
    -----
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
    Union[``float``, ``numpy.ndarray``]
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

def pearson_r(y_true, y_pred, multioutput="uniform", threshold=1.0) -> Union[float, np.ndarray]:
    """
    Compute the pearson correlation coefficient given two matrices y_true and y_pred.
    Each row is supposed to be an example (on which the pearson correlation coefficient) is computed. The computation
    happens independently for each column.

    Notes
    -----
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
    Union[float, np.ndarray]
        If `multioutput` is "uniform" it will return a floating point number corresponding to the average pearson
        correlation coefficient, otherwise it will return a vector with as many component as the number of columns
        in y_true an y_pred.

    """
    from scipy.stats import pearsonr

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

DEFAULT_METRICS = {"MSE_avg": mean_squared_error,
                   "MAE_avg": mean_absolute_error,
                   "NRMSE_avg": nrmse,
                   "pearson_r_avg": pearson_r,
                   "mape_avg": mape,
                   "rmse_avg": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)),
                   "mape_90_avg": mape_quantile,
                   "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, multioutput="raw_values"),
                   "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred, multioutput="raw_values"),
                   "NRMSE": lambda y_true, y_pred: nrmse(y_true, y_pred, multioutput="raw_values"),
                   "pearson_r": lambda y_true, y_pred: pearson_r(y_true, y_pred, multioutput="raw_values"),
                   "mape": lambda y_true, y_pred: mape(y_true, y_pred, multioutput="raw_values"),
                   "rmse": lambda y_true, y_pred: np.sqrt(
                       mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput="raw_values")),
                   "mape_90": lambda y_true, y_pred: mape_quantile(y_true=y_true,
                                                                   y_pred=y_pred,
                                                                   multioutput="raw_values"),
                   }