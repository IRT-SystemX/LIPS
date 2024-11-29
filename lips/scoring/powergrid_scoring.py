import math
from abc import ABC
from typing import Union, Dict

from lips.config import ConfigManager
from lips.logger import CustomLogger
from lips.scoring import Scoring
from lips.scoring.utils import read_json


class PowerGridScoring(Scoring, ABC):

    def __init__(self,
                 config: Union[ConfigManager, None] = None,
                 config_path: Union[str, None] = None,
                 scenario: Union[str, None] = None,
                 log_path: Union[str, None] = None
                 ):
        super().__init__(config=config,
                         config_path=config_path,
                         config_section=scenario,
                         log_path=log_path
                         )
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        self.thresholds = self.config.get_option("thresholds")
        self.coefficients = self.config.get_option("coefficients")
        self.value_by_color = self.config.get_option("valuebycolor")

    def scoring(self, metrics_path: str = "", metrics_dict: Union[Dict, str, None] = None):
        return read_json(json_path=metrics_path, json_object=metrics_dict)

    @staticmethod
    def calculate_score_color(metrics, thresholds):
        tree = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                tree[key] = PowerGridScoring.calculate_score_color(value, thresholds)
            else:
                discrete_metric = PowerGridScoring._discretize_metric(metric_name=key, metric_value=value,
                                                                      thresholds=thresholds)
                tree[key] = discrete_metric
        return tree

    @staticmethod
    def _discretize_metric(metric_name, metric_value, thresholds):
        """
        Discretize a metric value into a qualitative evaluation (g, o, r).

        :param metric_name: Name of the metric to evaluate
        :param metric_value: The value of the metric to be evaluated
        :param thresholds: Dictionary with thresholds for each metric. Format:
                           {
                               "metric_name": (threshold_min, threshold_max, eval_type)
                           }
                           eval_type can be "min" or "max".
        :return: Evaluation string ("g", "o", or "r")
        :raises ValueError: If the metric_name is not in thresholds or eval_type is invalid
        """
        # Ensure the metric name exists in thresholds
        if metric_name not in thresholds:
            available_metrics = ", ".join(thresholds.keys())
            raise ValueError(
                f"Metric '{metric_name}' not found in thresholds. Available metrics in thresholds: {available_metrics}")

        # Extract thresholds and evaluation type
        threshold_min, threshold_max, eval_type = thresholds[metric_name]

        # Validation for eval_type
        if eval_type not in {"min", "max"}:
            raise ValueError(f"Invalid eval_type '{eval_type}' for metric '{metric_name}'. Must be 'min' or 'max'.")

        # Determine evaluation based on thresholds and eval_type
        if eval_type == "min":
            # "min" means smaller values are better
            evaluation = "g" if metric_value <= threshold_min else "o" if metric_value < threshold_max else "r"
        else:
            # "max" means larger values are better
            evaluation = "r" if metric_value <= threshold_min else "o" if metric_value < threshold_max else "g"

        return evaluation

    @staticmethod
    def _calculate_speed_score(time_inference, time_classical_solver, max_speed_ratio_allowed):
        speed_up = PowerGridScoring._calculate_speed_up(time_classical_solver, time_inference)
        return max(min(math.log10(speed_up) / math.log10(max_speed_ratio_allowed), 1), 0)

    @staticmethod
    def _calculate_speed_up(time_classical_solver, time_inference):
        return time_classical_solver / time_inference
