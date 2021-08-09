import numpy as np
# mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lips.metrics import nrmse
from lips.metrics import pearson_r
from lips.metrics import mape

DEFAULT_METRICS = {"MSE_avg": mean_squared_error,
                   "MAE_avg": mean_absolute_error,
                   "NRMSE_avg": nrmse,
                   "pearson_r_avg": pearson_r,
                   "mape_avg": mape,
                   "rmse_avg": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)),
                   "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, multioutput="raw_values"),
                   "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred, multioutput="raw_values"),
                   "NRMSE": lambda y_true, y_pred: nrmse(y_true, y_pred, multioutput="raw_values"),
                   "pearson_r": lambda y_true, y_pred: pearson_r(y_true, y_pred, multioutput="raw_values"),
                   "mape": lambda y_true, y_pred: mape(y_true, y_pred, multioutput="raw_values"),
                   "rmse": lambda y_true, y_pred: np.sqrt(
                       mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput="raw_values")),
                   }
