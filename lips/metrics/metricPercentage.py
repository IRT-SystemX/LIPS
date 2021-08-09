import numpy as np
from collections.abc import Iterable
from lips.metrics.utils import DEFAULT_METRICS


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
