# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from ...evaluation.utils import metric_factory

try:
    from leap_net.metrics import mape_quantile
    metric_factory.register_metric("MAPE_90_avg", mape_quantile)
    metric_factory.register_metric("MAPE_90", lambda y_true, y_pred: mape_quantile(y_true=y_true,
                                                                                   y_pred=y_pred,
                                                                                   multioutput="raw_values"))
    metric_factory.register_metric("MAPE_10_avg", lambda y_true, y_pred: mape_quantile(y_true=y_true,
                                                                                       y_pred=y_pred,
                                                                                       quantile=0.9))
    metric_factory.register_metric("MAPE_10", lambda y_true, y_pred: mape_quantile(y_true=y_true,
                                                                                   y_pred=y_pred,
                                                                                   multioutput="raw_values",
                                                                                   quantile=0.9))
except ImportError as exc_:
    pass
