# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import numpy as np
from collections.abc import Iterable
from lips.metrics import DEFAULT_METRICS


def metricPercentage(metrics_ML, metric_names, observations, predictions, k, variables=["a_or", "p_or", "v_or"]):
    """
    take a metric and compute it on a percentage of data with highest values of voltages

    params
    ------
        k: ``float``
            indicate the proportion of highest values to be considered in metrics computations
    """
    if metric_names is None:
        metrics = DEFAULT_METRICS
    else:
        metrics = {nm_: DEFAULT_METRICS[nm_] for nm_ in metric_names}

    #dict_metrics = dict()
    k_percent = np.int(k * predictions["a_or"].size)
    indices = np.argsort(observations["a_or"].flatten())[-k_percent:]

    for metric_name, metric_fun in metrics.items():
        metric_name = metric_name + str(int(k*100))
        metrics_ML[metric_name] = {}
        for nm in variables:
            true_ = observations[nm]
            pred_ = predictions[nm]
            tmp = metric_fun(
                true_.flatten()[indices], pred_.flatten()[indices])
            if isinstance(tmp, Iterable):
                metrics_ML[metric_name][nm] = [
                    float(el) for el in tmp]
            else:
                metrics_ML[metric_name][nm] = float(tmp)

    # return metrics_ML
