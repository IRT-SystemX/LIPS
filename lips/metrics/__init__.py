"""
Usage:
    Init file for metrics module
Licence:
    Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

#__all__ = ["mape_quantile", "nrmse", "mape", "pearson_r", "DEFAULT_METRICS"]

#from lips.metrics.ml_metrics import mape_quantile, nrmse, mape, pearson_r
#from lips.metrics.ml_metrics import DEFAULT_METRICS

__all__ = ["DEFAULT_METRICS"]

try:
    from leap_net.metrics import nrmse
    from leap_net.metrics import mape
    from leap_net.metrics import mape_quantile
    from leap_net.metrics import pearson_r
    from leap_net.proxy.utils import DEFAULT_METRICS
    __all__ += ["nrmse", "mape", "mape_quantile", "pearson_r"]
except ImportError as exc_:
    # leap net package not available
    DEFAULT_METRICS = {}