# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0

import bisect
from abc import ABC
from typing import Union, List, Dict, Any

from ..config import ConfigManager
from ..logger import CustomLogger

# Constants
VALID_COMPARISONS = {"minimize", "maximize"}


class Scoring(ABC):
    def __init__(self, config: Union[ConfigManager, None] = None, config_path: Union[str, None] = None,
                 config_section: Union[str, None] = None, log_path: Union[str, None] = None):
        """
        Initializes the Scoring instance, loading configuration and setting up logger.

        Args:
            config (ConfigManager, optional): A ConfigManager instance. Defaults to None.
            config_path (str, optional): Path to the configuration file. Defaults to None.
            config_section (str, optional): Section of the configuration file. Defaults to None.
            log_path (str, optional): Path to the log file. Defaults to None.
        """
        self.config = config if config else ConfigManager(section_name=config_section, path=config_path)
        self.logger = CustomLogger(__class__.__name__, log_path).logger

        self.thresholds = self.config.get_option("thresholds")
        self.value_by_color = self.config.get_option("valuebycolor")

        self._validate_thresholds_config()

    def calculate_score_color(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively calculates the score color for each metric in a tree metrics dictionary.

        Args:
            metrics: A dictionary of metric names and their corresponding values.

        Returns:
            A dictionary with the same structure as `metrics` but with colorized values.
        """
        tree = {}

        for key, value in metrics.items():
            if isinstance(value, dict):
                tree[key] = self.calculate_score_color(value)
            else:
                threshold_data = self.thresholds[key]
                comparison_type = threshold_data["comparison_type"]
                metric_thresholds = threshold_data["thresholds"]
                discrete_metric = self.colorize_metric_value(metric_value=value, comparison_type=comparison_type,
                                                             thresholds=metric_thresholds)
                tree[key] = discrete_metric

        return tree

    def colorize_metric_value(self, metric_value: float, comparison_type: str, thresholds: List[float]) -> str:
        """
        Assigns a color based on the metric value, its comparison type (minimize or maximize), and thresholds.

        Args:
            metric_value (float): The value of the metric to be colorized.
            comparison_type (str): Either 'minimize' or 'maximize' to indicate how the threshold comparison is performed.
            thresholds (List[float]): A sorted list of threshold values.

        Returns:
            str: The color corresponding to the metric value based on its threshold position.

        Raises:
            ValueError: If the comparison_type is not 'minimize' or 'maximize'.
        """
        if comparison_type not in VALID_COMPARISONS:
            raise ValueError("comparison_type must be 'minimize' or 'maximize'")

        value_position = bisect.bisect_left(thresholds, metric_value)

        color_by_threshold = list(self.value_by_color.keys())

        return color_by_threshold[value_position] if comparison_type == "minimize" else color_by_threshold[
            - (value_position + 1)]

    def _validate_thresholds_config(self) -> None:
        """
        Validates the thresholds configuration against the value_by_color length.

        Raises:
            ValueError: If the thresholds configuration is invalid or missing required data.
        """
        if not self.thresholds:
            raise ValueError("Thresholds configuration is missing.")
        if not self.value_by_color:
            raise ValueError("Value by color configuration is missing.")

        expected_len = len(self.value_by_color) - 1
        for metric_name, threshold_data in self.thresholds.items():
            if not isinstance(threshold_data, dict) or "thresholds" not in threshold_data:
                raise ValueError(f"Invalid format for thresholds data for metric '{metric_name}'.")
            if len(threshold_data["thresholds"]) != expected_len:
                raise ValueError(
                    f"Metric '{metric_name}': Thresholds list length must be equal to ValueByColor length -1: i.e: {expected_len}.")
