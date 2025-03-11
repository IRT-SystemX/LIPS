# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking


import bisect
from abc import ABC
from typing import Union, List, Dict, Any

from ..config import ConfigManager
from ..logger import CustomLogger

# Constants
VALID_COMPARISONS = {"minimize", "maximize"}


class Scoring(ABC):
    """
    Base class for calculating scores based on metrics and thresholds.
    """

    def __init__(self, config: Union[ConfigManager, None] = None, config_path: Union[str, None] = None,
                 config_section: Union[str, None] = None, log_path: Union[str, None] = None):
        """
        Initializes the Scoring instance with configuration and logger.

        Args:
            config: A ConfigManager instance. Defaults to None.
            config_path: Path to the configuration file. Defaults to None.
            config_section: Section of the configuration file. Defaults to None.
            log_path: Path to the log file. Defaults to None.
        """
        self.config = config if config else ConfigManager(section_name=config_section, path=config_path)
        self.logger = CustomLogger(__class__.__name__, log_path).logger

        self.thresholds = self.config.get_option("thresholds")
        self.value_by_color = self.config.get_option("valuebycolor")
        self.coefficients = self.config.get_option("coefficients")
        self._validate_configuration()

    def colorize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively colorizes metric values based on thresholds.

        Args:
            metrics: A dictionary of metric names and their corresponding values (can be nested).

        Returns:
            A dictionary with the same structure as `metrics` but with colorized values.
        """
        colorized_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                colorized_metrics[key] = self.colorize_metrics(value)
            else:
                colorized_metrics[key] = self._colorize_metric_value(key, value)
        return colorized_metrics

    def _colorize_metric_value(self, metric_name: str, metric_value: float) -> str:
        """
        Assigns a color to a single metric value based on its threshold.

        Args:
            metric_name: The name of the metric.
            metric_value: The value of the metric.

        Returns:
            The color corresponding to the metric value.

        Raises:
            ValueError: If the comparison type is invalid.
        """
        threshold_data = self.thresholds[metric_name]
        comparison_type = threshold_data["comparison_type"]
        thresholds = threshold_data["thresholds"]

        if comparison_type not in VALID_COMPARISONS:
            raise ValueError(f"Invalid comparison type: {comparison_type}. Must be 'minimize' or 'maximize'.")

        index = bisect.bisect_left(thresholds, metric_value)
        colors = list(self.value_by_color.keys())
        return colors[index] if comparison_type == "minimize" else colors[-(index + 1)]

    def _validate_configuration(self) -> None:
        """
        Validates the thresholds and value_by_color configurations.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.thresholds:
            raise ValueError("Thresholds configuration is missing.")
        if not self.value_by_color:
            raise ValueError("Value by color configuration is missing.")

        expected_threshold_count = len(self.value_by_color) - 1
        for metric_name, threshold_data in self.thresholds.items():
            if not isinstance(threshold_data,
                              dict) or "thresholds" not in threshold_data or "comparison_type" not in threshold_data:
                raise ValueError(
                    f"Invalid thresholds data for metric '{metric_name}'. Must be a dict with 'thresholds' and 'comparison_type' keys.")
            if (threshold_data["comparison_type"] in VALID_COMPARISONS) and (
                    len(threshold_data["thresholds"]) != expected_threshold_count):
                raise ValueError(
                    f"Metric '{metric_name}': Thresholds count must be {expected_threshold_count} (length of ValueByColor - 1).")

    def _calculate_leaf_score(self, colors: List[str]) -> float:
        """
        Calculates the score for a leaf node (set of colorized metrics).

        Args:
            colors: A list of color strings representing the colorized metrics.

        Returns:
            The calculated score for the leaf node.
        """
        return sum(self.value_by_color[color] for color in colors) / (len(colors) * max(self.value_by_color.values()))

    def calculate_sub_scores(self, node: Dict[str, Any]) -> Union[float, Dict[str, Any]]:
        """
        Calculates sub-scores recursively for a node in the metrics tree.

        Args:
            node: A node in the metrics tree (can be a leaf or a sub-tree).

        Returns:
            The sub-score for the node (float for leaf, dict for sub-tree).

        Raises:
            ValueError: If the input JSON is not a dictionary or if a parent node is inconsistently branched.
        """
        if not isinstance(node, dict):
            raise ValueError("Input must be a dictionary.")

        if all(isinstance(value, str) for value in node.values()):  # Leaf node
            return self._calculate_leaf_score(list(node.values()))
        elif any(isinstance(value, str) for value in node.values()):  # Inconsistent branching
            raise ValueError("Parent node is not uniformly branched (mix of leaf and sub-tree children).")
        else:  # Sub-tree node
            return {key: self.calculate_sub_scores(value) for key, value in node.items()}

    def calculate_global_score(self, tree: Union[float, Dict[str, Any]], key_path: List[str] = None) -> float:
        """
        Calculates the global score for the entire metrics tree.

        Args:
            tree: a pre-calculated sub-score tree
            key_path: the path to the current node in the tree

        Returns:
            The global score.

        """

        if isinstance(tree, (int, float)):  # Base case: already a sub-score
            return tree

        key_path = key_path or []
        global_score = 0
        for key, subtree in tree.items():
            new_path = key_path + [key]

            weight = self._get_coefficient(new_path) or 1  # Default weight is 1
            global_score += weight * self.calculate_global_score(subtree, new_path)

        return global_score

    def _get_coefficient(self, key_path: List[str]) -> Union[float, None]:
        """
        Retrieves the coefficient value from a nested dictionary based on a given path.
        Args:
            key_path: A list of keys representing the path to the desired coefficient.

        Returns:
            The coefficient value if found, otherwise None.
        """
        current = self.coefficients
        for key in key_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                self.logger.warning(f"Coefficient not found for path: {' -> '.join(key_path)}. Using default value 1.")
                return None
        return current.get("value")
