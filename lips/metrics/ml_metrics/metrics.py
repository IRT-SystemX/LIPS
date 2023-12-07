# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from ...evaluation.utils import metric_factory

from .normalized_mean_squared_error import normalized_mean_squared_error
from .mean_squared_error import mean_squared_error 
from .mean_absolute_error import mean_absolute_error 
from .mean_absolute_percentage_error import mean_absolute_percentage_error

metric_factory.register_metric("MAPE_avg", mean_absolute_percentage_error)
metric_factory.register_metric("MAPE", lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred, output="raw"))
metric_factory.register_metric("MAE_avg", mean_absolute_error)
metric_factory.register_metric("MAE", lambda y_true, y_pred: mean_absolute_error(y_true, y_pred, output="raw"))
metric_factory.register_metric("MSE_avg", mean_squared_error)
metric_factory.register_metric("MSE", lambda y_true, y_pred: mean_squared_error(y_true, y_pred, output="raw"))
metric_factory.register_metric("RMSE_avg", lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False))
metric_factory.register_metric("RMSE", lambda y_true, y_pred: mean_squared_error(y_true, y_pred, output="raw", squared=False))
metric_factory.register_metric("NRMSE_avg", normalized_mean_squared_error)
metric_factory.register_metric("NRMSE", lambda y_true, y_pred: normalized_mean_squared_error(y_true, y_pred, output="raw"))
