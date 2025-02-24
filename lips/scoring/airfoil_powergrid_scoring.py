import math
from typing import Union, Dict, List

from .scoring import Scoring
from .utils import get_nested_value, filter_metrics, read_json
from ..config import ConfigManager


class AirfoilPowerGridScoring(Scoring):
    """
    Class responsible for calculating the score of the AirFoil Power Grid competition : https://www.codabench.org/competitions/3282/
    """

    def __init__(self, config: Union[ConfigManager, None] = None, config_path: Union[str, None] = None,
                 config_section: Union[str, None] = None, log_path: Union[str, None] = None):
        """
        Initializes the AirfoilPowerGridScoring instance with configuration and logger.

        Args:
            config: A ConfigManager instance. Defaults to None.
            config_path: Path to the configuration file. Defaults to None.
            config_section: Section of the configuration file. Defaults to None.
            log_path: Path to the log file. Defaults to None.
        """
        super().__init__(config=config, config_path=config_path, config_section=config_section, log_path=log_path)

    def _reconstruct_ml_metrics(self, raw_metrics: Dict, ml_key_path: List[str]) -> Dict:
        """
        Construct ML metrics by retrieving and filtering data from the raw-JSON metrics.

        Args:
            raw_metrics: Dictionary containing the raw metrics data.
            ml_key_path: List of keys representing the path to the ML metrics
                         within the raw_metrics dictionary.

        Returns:
            Dictionary containing the filtered ML metrics.

        Raises:
            ValueError: If the specified path is invalid or ML metrics are not found.
            TypeError: If the value at the specified path is not a dictionary.
        """

        all_ml_metrics = get_nested_value(raw_metrics, ml_key_path)
        if all_ml_metrics is None:
            raise ValueError(f"Invalid path {ml_key_path}. Could not retrieve ML metrics.")

        if not isinstance(all_ml_metrics, dict):
            raise TypeError(f"Expected a dictionary at {ml_key_path}, but got {type(all_ml_metrics).__name__}.")

        ml_metrics = {"ML": filter_metrics(all_ml_metrics, self.thresholds.keys())}

        pressure_surfacic_value_path = ml_key_path[:-1] + ["MSE_normalized_surfacic", "pressure"]
        ml_metrics["ML"]["pressure_surfacic"] = get_nested_value(raw_metrics, pressure_surfacic_value_path)

        return ml_metrics

    def _reconstruct_physic_metrics(self, raw_metrics: Dict, physic_key_path: List[str]) -> Dict:
        """
        Construct Physic metrics by retrieving and filtering data from the raw-JSON metrics .

        Args:
            raw_metrics: Dictionary containing the raw metrics data.
            physic_key_path: List of keys representing the path to the physics metrics
                             within the raw_metrics dictionary.

        Returns:
            Dictionary containing the filtered physics metrics.

        Raises:
            ValueError: If the specified path is invalid or physics metrics are not found.
            TypeError: If the value at the specified path is not a dictionary.
        """

        all_physic_metrics = get_nested_value(raw_metrics, physic_key_path)
        if all_physic_metrics is None:
            raise ValueError(f"Invalid path {physic_key_path}. Could not retrieve Physic metrics.")

        if not isinstance(all_physic_metrics, dict):
            raise TypeError(f"Expected a dictionary at {physic_key_path}, but got {type(all_physic_metrics).__name__}.")

        physic_metrics = {"Physics": filter_metrics(all_physic_metrics, self.thresholds.keys())}
        return physic_metrics

    def _reconstruct_ood_metrics(self, raw_metrics: Dict, ml_ood_key_path: List[str],
                                 physic_ood_key_path: List[str]) -> Dict:
        """
        Construct OOD metrics by retrieving and Combining ML and Physic OOD-metrics from the raw-JSON metrics .

        Args:
            raw_metrics: Dictionary containing the raw metrics data.
            ml_ood_key_path: Path to the ML OOD metrics.
            physic_ood_key_path: Path to the Physics OOD metrics.

        Returns:
            Dictionary containing the combined OOD metrics.
        """
        ml_ood_metrics = self._reconstruct_ml_metrics(raw_metrics, ml_ood_key_path)["ML"]
        physic_ood_metrics = self._reconstruct_physic_metrics(raw_metrics, physic_ood_key_path)["Physics"]

        return {"OOD": {**ml_ood_metrics, **physic_ood_metrics}}

    def compute_speed_score(self, time_inference: float) -> float:
        """
        Computes the speed score based on:

        Score_Speed = min( (log10(SpeedUp) / log10(SpeedUpMax)), 1)

        Where : SpeedUp = time_ClassicalSolver / time_Inference

        Args:
            time_inference: Inference time in seconds.

        Returns:
            The speed score (between 0 and 1).
        """

        speed_up = self._calculate_speed_up(time_inference)
        max_speed_ratio_allowed = self.thresholds["max_speed_ratio_allowed"]["thresholds"][0]
        res = min(math.log10(speed_up) / math.log10(max_speed_ratio_allowed), 1)
        return max(res, 0)

    def _calculate_speed_up(self, time_inference: float) -> float:
        """Calculates the speedup factor based on:
        SpeedUp = time_ClassicalSolver / time_Inference
        """

        time_classical_solver = self.thresholds["reference_mean_simulation_time"]["thresholds"][0]
        return time_classical_solver / time_inference

    def compute_scores(self, metrics_dict: Union[Dict, None] = None, metrics_path: str = "") -> Dict:
        """
        Computes the competition score based on the provided metrics in metrics_dict or metrics_path

        Args:
            metrics_dict: Dictionary containing the raw metrics data.
            metrics_path: Path to the JSON file containing the raw metrics data.

        Returns:
            Dictionary containing the score colors, the score values, and the global score.
        Raises:
            ValueError: If both metrics_dict and metrics_path are None.
        """
        
        if metrics_dict is not None:
            metrics = metrics_dict.copy()
        elif metrics_path != "":
            metrics = read_json(json_path=metrics_path, json_object=metrics_dict)
        else:
            raise ValueError("metrics_path and metrics_dict cant' both be None")

        time_inference = metrics_dict["test_mean_simulation_time"]

        ml_metrics = self._reconstruct_ml_metrics(metrics,
                                                  ml_key_path=["fc_metrics_test", "test", "ML", "MSE_normalized"])
        physic_metrics = self._reconstruct_physic_metrics(metrics,
                                                          physic_key_path=["fc_metrics_test", "test", "Physics"])
        ood_metrics = self._reconstruct_ood_metrics(metrics, ml_ood_key_path=['fc_metrics_test_ood', 'test_ood', 'ML',
                                                                              'MSE_normalized'],
                                                    physic_ood_key_path=['fc_metrics_test_ood', 'test_ood', 'Physics'])

        metrics = {**ml_metrics, **physic_metrics, **ood_metrics}

        sub_scores_color = self.colorize_metrics(metrics)
        sub_scores_values = self.calculate_sub_scores(sub_scores_color)

        speed_score = self.compute_speed_score(time_inference)
        sub_scores_values["ML"] = {"Accuracy": sub_scores_values["ML"], "SpeedUP": speed_score}
        sub_scores_values["OOD"] = {"Accuracy": sub_scores_values["OOD"], "SpeedUP": speed_score}

        global_score = self.calculate_global_score(sub_scores_values)

        return {"Score Colors": sub_scores_color, "Score values": sub_scores_values, "Global Score": global_score}
