# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from abc import abstractmethod
from typing import Union, Dict

from lips.config import ConfigManager
from lips.scoring import Scoring


class PowerGridScoring(Scoring):
    """
    Abstract base class for calculating scores for power grid use cases.
    """

    def __init__(self, config: Union[ConfigManager, None] = None, config_path: Union[str, None] = None,
                 config_section: Union[str, None] = None, log_path: Union[str, None] = None):
        """
        Initializes the PowerGridScoring instance with configuration and logger.

        Args:
            config: A ConfigManager instance. Defaults to None.
            config_path: Path to the configuration file. Defaults to None.
            config_section: Section of the configuration file. Defaults to None.
            log_path: Path to the log file. Defaults to None.
        """
        super().__init__(config=config, config_path=config_path, config_section=config_section, log_path=log_path)

    @abstractmethod
    def _reconstruct_ml_metrics(self, **kwargs) -> Dict:
        """
        Reconstructs the ML metrics from the raw data.

        Returns:
            A dictionary containing the reconstructed ML metrics.
        """
        pass

    @abstractmethod
    def _reconstruct_physic_metrics(self, **kwargs) -> Dict:
        """
        Reconstructs the physics metrics from the raw data.

        Returns:
            A dictionary containing the reconstructed physics metrics.
        """
        pass

    @abstractmethod
    def _reconstruct_ood_metrics(self, **kwargs) -> Dict:
        """
        Reconstructs the OOD metrics from the raw data.

        Returns:
            A dictionary containing the reconstructed OOD metrics.
        """
        pass

    @abstractmethod
    def compute_scores(self, metrics_dict: Union[Dict, None] = None, metrics_path: str = "") -> Dict:
        """
        Computes the scores based on the provided metrics.

        Args:
            metrics_dict: A dictionary containing the metrics data.
            metrics_path: The path to a JSON file containing the metrics data.

        Returns:
            A dictionary containing the calculated scores.
        """
        pass

    @abstractmethod
    def compute_speed_score(self, time_inference: float) -> float:
        """
        Computes the speed score based on the inference time.

        Args:
            time_inference: The inference time.

        Returns:
            The calculated speed score.
        """
        pass

    def _calculate_speed_up(self, time_inference: float) -> float:
        """
        Calculates the speedup factor based on:
           SpeedUp = time_ClassicalSolver / time_Inference
        Args:
            time_inference: The inference time in seconds.

        Returns:
            The calculated speedup factor.
        """

        time_classical_solver = self.thresholds["reference_mean_simulation_time"]["thresholds"][0]
        return time_classical_solver / time_inference
