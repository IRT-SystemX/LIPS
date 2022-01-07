# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import copy
from typing import Union
import numpy as np

from lips.metrics import DEFAULT_METRICS
from lips.logger import CustomLogger

#logging.basicConfig(filename="logs.log",
#                    level=logging.INFO,
#                    format="%(levelname)s:%(message)s")


def metricPercentage(metrics_ML,  
                     observations, 
                     predictions, 
                     k=0.1, 
                     metric_names=["mape", "MAE"], 
                     variables=["a_or", "p_or", "v_or"], 
                     agg_func=np.mean,
                     log_path: Union[str, None]=None
                     ):
    """
    take a metric and compute it on a percentage of data with highest values of voltages

    params
    ------
        metrics_ML: ``dict``
            a dictionary containing the metric key, value pair

        observations: ``dict``
            the real observations. a dictionary with keys presenting the variable names

        predictions: ``dict``
            the predicted observations. a dictionary with keys presenting the variable names

        k: ``float``
            indicate the proportion of electrical current (AMP) highest values to be considered in metrics computations

        metric_names: ``list``
            a list of metrics to be computed on k% of observations with highest current values. If it is none, all the metrics
            are considered. Attention : not all the metrics all compatible with this computation

        variables: ``list``
            a list of variables for which the metrics shall be computed

        agg_func: ``numpy function``
            the aggregate function indicate how the final results should be aggregated once computed per line
            it could be ``np.sum``, ``np.mean``, ``np.median`` or any other appropriate aggregation function

        
    """
    # logger  
    logger = CustomLogger("metric_percentage", log_path).logger

    FLOW_VARIABLES = ['a_or', 'a_ex', 'p_or', 'p_ex', 'q_or', 'q_ex', 'v_or', 'v_ex', 'theta_or', 'theta_ex']
    variables = [var for var in variables if var in FLOW_VARIABLES]


    metrics_ML_raw = dict()

    if metric_names is None:
        metrics = DEFAULT_METRICS
    else:
        metrics = {nm_: DEFAULT_METRICS[nm_] for nm_ in metric_names}

    n, p = observations["a_or"].shape
    k_percent = np.int(k * n)

    for metric_name, metric_func in metrics.items():
        metric_name = metric_name + str(int(100 - k*100))
        metrics_ML_raw[metric_name] = {}
        metrics_ML[metric_name] = {}
        # compute on desired variables
        for var_ in variables:
            metrics_ML_raw[metric_name][var_] = list()
            # compute the metrics on 10% of samples per line presenting highest current values
            for l in range(p):
                # indices of k percent of highest values
                indices = np.argsort(observations["a_or"][:,l])[-k_percent:]
                true_ = observations[var_][:,l][indices]
                pred_ = predictions[var_][:,l][indices]
                tmp = metric_func(true_, pred_)
                metrics_ML_raw[metric_name][var_].append(tmp)
            metrics_ML[metric_name][var_] = float(agg_func(metrics_ML_raw[metric_name][var_]))
            logger.info("{} for {} is : {:.3f}".format(metric_name, var_, float(agg_func(metrics_ML_raw[metric_name][var_]))))

    return copy.deepcopy(metrics_ML), copy.deepcopy(metrics_ML_raw)
