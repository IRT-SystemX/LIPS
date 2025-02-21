import math
from typing import Union, Dict, List

from lips.config import ConfigManager
from lips.logger import CustomLogger
from lips.scoring import Scoring
from lips.scoring import utils
from lips.scoring.utils import get_nested_value, filter_metrics


class PowerGridScoring(Scoring):

    def __init__(self, config: Union[ConfigManager, None] = None, config_path: Union[str, None] = None,
                 scenario: Union[str, None] = None, log_path: Union[str, None] = None):
        super().__init__(config=config, config_path=config_path, config_section=scenario, log_path=log_path)
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        self.thresholds = self.config.get_option("thresholds")
        self.coefficients = self.config.get_option("coefficients")
        self.value_by_color = self.config.get_option("valuebycolor")

        self.speed_config = self.config.get_option("speedconfig")

    def scoring(self, metrics_path: str = "", metrics_dict: Union[Dict, str, None] = None):

        if metrics_dict is not None:
            metrics = metrics_dict.copy()
        elif metrics_path != "":
            metrics = utils.read_json(json_path=metrics_path, json_object=metrics_dict)
        else:
            raise ValueError("metrics_path and metrics_dict cant' both be None")

        # calculate speed score
        time_inference = metrics.pop("Speed")["inference_time"]
        speed_score = self._calculate_speed_score(time_inference)
        # score discretize
        score_color = PowerGridScoring._calculate_score_color(metrics, self.thresholds)

        score_values = dict()

        for key in self.coefficients:
            if key in score_color:
                flat_dict = utils.flatten_dict(score_color[key])
                score_values[key] = self._calculate_leaf_score(flat_dict.values())
        score_values["Speed"] = speed_score

        # calculate global score value
        global_score = self._calculate_global_score(score_values)
        score_values["Global Score"] = global_score

        return {"Score Colors": score_color, "Score Values": score_values}

    @staticmethod
    def _calculate_score_color(metrics, thresholds):
        tree = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                tree[key] = PowerGridScoring._calculate_score_color(value, thresholds)
            else:

                discrete_metric = PowerGridScoring._discretize_metric(metric_name=key, metric_value=value,
                                                                      thresholds=thresholds)
                tree[key] = discrete_metric
        return tree

    def _calculate_leaf_score(self, colors: List[str]):
        s = sum([self.value_by_color[color] for color in colors])
        return s / (len(colors) * max(self.value_by_color.values()))

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

    # separate competions from powergrid_scoring
    def _calculate_speed_score(self, time_inference):

        time_classical_solver = self.speed_config["reference_mean_simulation_time"]
        speed_up = PowerGridScoring._calculate_speed_up(time_classical_solver, time_inference)

        if self.speed_config["speed_score_formula"] == "PowerGrid Competition":
            return PowerGridScoring._calculate_speed_score_powergrid_competition_formula(speed_up)
        elif self.speed_config["speed_score_formula"] == "AirFoil Competition":

            max_speed_ratio_allowed = self.speed_config["max_speed_ratio_allowed"]

            return PowerGridScoring._calculate_speed_score_airfoil_competition_formula(speed_up,
                                                                                       max_speed_ratio_allowed)
        else:
            raise ValueError(f'{self.speed_config["speed_score_formula"]} formula not found, please implement it first')

    @staticmethod
    def _calculate_speed_up(time_classical_solver, time_inference):
        return time_classical_solver / time_inference

    @staticmethod
    def _calculate_speed_score_airfoil_competition_formula(speed_up, max_speed_ratio_allowed):
        res = min(math.log10(speed_up) / math.log10(max_speed_ratio_allowed), 1)
        return max(res, 0)

    @staticmethod
    def _calculate_speed_score_powergrid_competition_formula(speed_up):
        res = utils.weibull(5, 1.7, speed_up)
        return max(min(res, 1), 0)

    def _calculate_global_score(self, sub_scores):
        global_score = 0
        for coef in self.coefficients.keys():
            global_score += self.coefficients[coef] * sub_scores[coef]
        return global_score

    @staticmethod
    def reconstruct_ml_metrics(input_json, ml_key_path, used_metric_list):
        """
        Construct ML metrics by retrieving and filtering data from the given JSON.

        Parameters:
        - input_json (dict): The input JSON containing the ML metrics.
        - ml_key_path (list): Path to the ML section in the JSON as a list of keys.
        - used_metric_list (list): List of metrics to include in the output.

        Returns:
        - dict: Filtered ML metrics containing only the specified metrics.
        """
        all_ml_metrics = get_nested_value(input_json, ml_key_path)
        if all_ml_metrics is None:
            raise ValueError(f"Invalid path {ml_key_path}. Could not retrieve ML metrics.")

        if not isinstance(all_ml_metrics, dict):
            raise TypeError(f"Expected a dictionary at {ml_key_path}, but got {type(all_ml_metrics).__name__}.")

        return {"ML": filter_metrics(all_ml_metrics, used_metric_list)}

    @staticmethod
    def reconstruct_speed_metric(input_json, speed_key_path):
        """
        Construct a dictionary containing the speed metric.

        Parameters:
        - input_json (dict): The input JSON containing the speed metric.
        - speed_key_path (list): Path to the inference time in the JSON as a list of keys.

        Returns:
        - dict: A dictionary with the speed metric in the format {"Speed": {"inference_time": value}}.

        Raises:
        - ValueError: If the specified path does not exist or the value is None.
        """
        inference_time = get_nested_value(input_json, speed_key_path)

        if inference_time is None:
            raise ValueError(f"Invalid path {speed_key_path}. Could not retrieve inference time.")

        if not isinstance(inference_time, (int, float)):
            raise TypeError(f"Inference time must be a numeric value, but got {type(inference_time).__name__}.")

        return {"Speed": {"inference_time": inference_time}}

    @staticmethod
    def reconstruct_physic_metrics(input_json, physic_key_path, competition_name, used_metric_list=None):

        all_physic_metrics = get_nested_value(input_json, physic_key_path)
        if all_physic_metrics is None:
            raise ValueError(f"Invalid path {physic_key_path}. Could not retrieve physic metrics.")

        if not isinstance(all_physic_metrics, dict):
            raise TypeError(f"Expected a dictionary at {physic_key_path}, but got {type(all_physic_metrics).__name__}.")

        if competition_name == "AirFoil Competition":
            physic_metrics = filter_metrics(all_physic_metrics, used_metric_list)

        elif competition_name == "PowerGrid Competition":
            physic_metrics = {"CURRENT_POS": all_physic_metrics["CURRENT_POS"]["a_or"]["Violation_proportion"] * 100.,
                              "VOLTAGE_POS": all_physic_metrics["VOLTAGE_POS"]["v_or"]["Violation_proportion"] * 100.,
                              "LOSS_POS": all_physic_metrics["LOSS_POS"]["violation_proportion"] * 100.,
                              "DISC_LINES": all_physic_metrics["DISC_LINES"]["violation_proportion"] * 100.,
                              "CHECK_LOSS": all_physic_metrics["CHECK_LOSS"]["violation_percentage"],
                              "CHECK_GC": all_physic_metrics["CHECK_GC"]["violation_percentage"],
                              "CHECK_LC": all_physic_metrics["CHECK_LC"]["violation_percentage"],
                              "CHECK_JOULE_LAW": all_physic_metrics["CHECK_JOULE_LAW"]["violation_proportion"] * 100.}

        else:
            raise ValueError(f'{competition_name} not in options')

        return {"Physics": physic_metrics}

