# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import math
from typing import Union, Dict, List

from .powergrid_scoring import PowerGridScoring
from .utils import get_nested_value, read_json
from ..config import ConfigManager


class ML4PhysimPowerGridScoring(PowerGridScoring):
    """
    Calculates the score for the ML4Physim Power Grid competition: https://www.codabench.org/competitions/2378/
    """

    def __init__(self, config: Union[ConfigManager, None] = None, config_path: Union[str, None] = None,
                 config_section: Union[str, None] = None, log_path: Union[str, None] = None):
        """
        Initializes the ML4PhysimPowerGridScoring instance with configuration and logger.

        Args:
            config: A ConfigManager instance. Defaults to None.
            config_path: Path to the configuration file. Defaults to None.
            config_section: Section of the configuration file. Defaults to None.
            log_path: Path to the log file. Defaults to None.
        """
        super().__init__(config=config, config_path=config_path, config_section=config_section, log_path=log_path)

    def _reconstruct_ml_metrics(self, raw_metrics: Dict, ml_section_path: List[str]) -> Dict:
        """
        Construct ML metrics by retrieving data from the raw-JSON metrics.

        Args:
            raw_metrics: Dictionary containing the raw metrics data.
            ml_section_path: List of keys representing the path to the ML section
                         within the raw_metrics dictionary.

        Returns:
            Dictionary containing the desired ML metrics.

        Raises:
            ValueError: If the specified path is invalid or ML metrics are not found.
            TypeError: If the value at the specified path is not a dictionary.
        """

        ml_section = get_nested_value(raw_metrics, ml_section_path)

        if ml_section is None:
            raise ValueError(f"Invalid path {ml_section_path}. Could not retrieve ML metrics.")

        if not isinstance(ml_section, dict):
            raise TypeError(f"Expected a dictionary at {ml_section_path}, but got {type(ml_section).__name__}.")

        ml_metrics = {}

        ml_metrics["a_or"] = ml_section["MAPE_90_avg"]["a_or"]
        ml_metrics["a_ex"] = ml_section["MAPE_90_avg"]["a_ex"]
        ml_metrics["p_or"] = ml_section["MAPE_10_avg"]["p_or"]
        ml_metrics["p_ex"] = ml_section["MAPE_10_avg"]["p_ex"]
        ml_metrics["v_or"] = ml_section["MAE_avg"]["v_or"]
        ml_metrics["v_ex"] = ml_section["MAE_avg"]["v_ex"]

        return {"ML": ml_metrics}

    def _reconstruct_physic_metrics(self, raw_metrics: Dict, physic_section_path: List[str]) -> Dict:
        """
        Construct Physic metrics by retrieving and filtering data from the raw-JSON metrics .

        Args:
            raw_metrics: Dictionary containing the raw metrics data.
            physic_section_path: List of keys representing the path to the physics section
                             within the raw_metrics dictionary.

        Returns:
            Dictionary containing the filtered physics metrics.

        Raises:
            ValueError: If the specified path is invalid or physics metrics are not found.
            TypeError: If the value at the specified path is not a dictionary.
        """

        physic_section = get_nested_value(raw_metrics, physic_section_path)

        if physic_section is None:
            raise ValueError(f"Invalid path {physic_section_path}. Could not retrieve Physic metrics.")

        if not isinstance(physic_section, dict):
            raise TypeError(f"Expected a dictionary at {physic_section_path}, but got {type(physic_section).__name__}.")

        physic_metrics = {}

        physic_metrics["CURRENT_POS"] = physic_section["CURRENT_POS"]["a_or"]["Violation_proportion"] * 100.
        physic_metrics["VOLTAGE_POS"] = physic_section["VOLTAGE_POS"]["v_or"]["Violation_proportion"] * 100.
        physic_metrics["LOSS_POS"] = physic_section["LOSS_POS"]["violation_proportion"] * 100.
        physic_metrics["DISC_LINES"] = physic_section["DISC_LINES"]["violation_proportion"] * 100.
        physic_metrics["CHECK_LOSS"] = physic_section["CHECK_LOSS"]["violation_percentage"]
        physic_metrics["CHECK_GC"] = physic_section["CHECK_GC"]["violation_percentage"]
        physic_metrics["CHECK_LC"] = physic_section["CHECK_LC"]["violation_percentage"]
        physic_metrics["CHECK_JOULE_LAW"] = physic_section["CHECK_JOULE_LAW"]["violation_proportion"] * 100.

        return {"Physics": physic_metrics}

    def _reconstruct_ood_metrics(self, raw_metrics: Dict, ml_ood_section_path: List[str],
                                 physic_ood_section_path: List[str]) -> Dict:
        """
        Construct OOD metrics by retrieving and Combining ML and Physic OOD-metrics from the raw-JSON metrics .

        Args:
            raw_metrics: Dictionary containing the raw metrics data.
            ml_ood_section_path: Path to the ML OOD section.
            physic_ood_section_path: Path to the Physics OOD section.

        Returns:
            Dictionary containing the combined OOD metrics.
        """
        ml_ood_metrics = self._reconstruct_ml_metrics(raw_metrics, ml_ood_section_path)
        physic_ood_metrics = self._reconstruct_physic_metrics(raw_metrics, physic_ood_section_path)

        return {"OOD": {**ml_ood_metrics, **physic_ood_metrics}}

    def compute_speed_score(self, time_inference: float) -> float:
        """
        Computes the speed score based on:

        Score_Speed = min( weibull(SpeedUp), 1)

        Where : SpeedUp = time_ClassicalSolver / time_Inference

        Args:
            time_inference: Inference time in seconds.

        Returns:
            The speed score (between 0 and 1).
        """
        speed_up = self._calculate_speed_up(time_inference)
        res = min(self._weibull(5, 1.7, speed_up), 1)
        return max(res, 0)

    def _weibull(self, c, b, x):
        a = c * ((-math.log(0.9)) ** (-1 / b))
        return 1. - math.exp(-(x / a) ** b)

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

        time_inference = metrics["test"]["ML"]["TIME_INF"]

        ml_metrics = self._reconstruct_ml_metrics(metrics, ["test", "ML"])
        physic_metrics = self._reconstruct_physic_metrics(metrics, ["test", "Physics"])
        ood_metrics = self._reconstruct_ood_metrics(metrics, ["test_ood_topo", "ML"], ["test_ood_topo", "Physics"])

        metrics = {"ID": {**ml_metrics, **physic_metrics}, **ood_metrics}
        sub_scores_color = self.colorize_metrics(metrics)

        sub_scores_values = self.calculate_sub_scores(sub_scores_color)

        speed_score = self.compute_speed_score(time_inference)
        sub_scores_values["SpeedUP"] = speed_score

        global_score = self.calculate_global_score(sub_scores_values)

        return {"Score Colors": sub_scores_color, "Score values": sub_scores_values, "Global Score": global_score}
