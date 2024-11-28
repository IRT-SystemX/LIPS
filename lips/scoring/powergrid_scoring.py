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
        ##return read_json(json_path=metrics_path, json_object=metrics_dict)
        pass

    def _sub_soring(self):
        pass
