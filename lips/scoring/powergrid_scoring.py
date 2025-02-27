from abc import abstractmethod
from typing import Union, Dict

from lips.config import ConfigManager
from lips.scoring import Scoring


class PowerGridScoring(Scoring):

    def __init__(self, config: Union[ConfigManager, None] = None, config_path: Union[str, None] = None,
                 config_section: Union[str, None] = None, log_path: Union[str, None] = None):
        super().__init__(config=config, config_path=config_path, config_section=config_section, log_path=log_path)

    @abstractmethod
    def _reconstruct_ml_metrics(self, **kwargs) -> Dict:
        pass

    @abstractmethod
    def _reconstruct_physic_metrics(self, **kwargs) -> Dict:
        pass

    @abstractmethod
    def _reconstruct_ood_metrics(self, **kwargs) -> Dict:
        pass

    @abstractmethod
    def compute_scores(self, metrics_dict: Union[Dict, None] = None, metrics_path: str = "") -> Dict:
        pass

    @abstractmethod
    def compute_speed_score(self, time_inference: float) -> float:
        pass

    def _calculate_speed_up(self, time_inference: float) -> float:
        """Calculates the speedup factor based on:
        SpeedUp = time_ClassicalSolver / time_Inference
        """
        time_classical_solver = self.thresholds["reference_mean_simulation_time"]["thresholds"][0]
        return time_classical_solver / time_inference
