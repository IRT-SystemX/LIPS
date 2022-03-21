"""
Usage:
    PowerGridEvaluation allows to evaluate the augmented simulators from power grid context
Licence:
    Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

from typing import Union
from collections.abc import Iterable
from pprint import pprint # TODO: to remove (for debugging purpose)

from .evaluation import Evaluation
from ..logger import CustomLogger
from ..config import ConfigManager

class PowerGridEvaluation(Evaluation):
    """
    This class allows the evaluation of augmented simulators applied to power grids cases
    """
    def __init__(self,
                 observations: Union[dict, None]=None,
                 predictions: Union[dict, None]=None,
                 config_path: Union[str, None]=None,
                 config_section: Union[str, None]=None,
                 log_path: Union[str, None]=None
                 ):
        super().__init__(observations=observations,
                         predictions=predictions,
                         config_path=config_path,
                         config_section=config_section,
                         log_path=log_path
                         )
        if self.config_section:
            self.eval_dict = self.config.get_option("eval_dict")
            self.eval_params = self.config.get_option("eval_params")
        else:
            # load the default section
            self.config = ConfigManager(benchmark_name="DEFAULT", path=self.config_path)
            self.eval_dict = self.config.get_option("eval_dict")
            self.eval_params = self.config.get_option("eval_params")
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        # read the criteria and their mapped functions for power grid
        self.criteria = self.mapper.map_powergrid_criteria()
        
    @classmethod
    def from_benchmark(cls, benchmark, config=None, log_path=None):
        """
        Initialize the class from benchmark object
        """
        return cls(benchmark.observations, benchmark.predictions, config, log_path)

    def evaluate(self, save_path: Union[str, None]=None):
        """
        The main function which evaluates all the required criteria noted in config file
        """
        # call the base class for generic evaluations
        super().evaluate()

        # evaluate powergrid specific evaluations based on config
        for cat in self.eval_dict.keys():
            self._dispatch_evaluation(cat)
                
        # TODO: save the self.metrics variable
        if save_path:
            pass

    def _dispatch_evaluation(self, category: str):
        """
        This helper function select the evaluation function with respect to the category

        params
        ======
        category: `str`
            the evaluation criteria category, the values could be one of the [`ML`, `Physics`, `IndRed`, `OOD`]
        """
        if category == self.MACHINE_LEARNING:
            # verify if the list is not empty
            if self.eval_dict[category]:
                self.evaluate_ml()
        if category == self.PHYSICS_COMPLIANCES:
            if self.eval_dict[category]:
                self.evaluate_physics()
        if category == self.INDUSTRIAL_READINESS:
            if self.eval_dict[category]:
                self.evaluate_industrial_readiness()
        if category == self.OOD_GENERALIZATION:
            if self.eval_dict[category]:
                self.evaluate_ood()

    def evaluate_ml(self):
        """
        Verify PowerGrid Specific Machine Learning metrics such as MAPE90
        """
        metric_dict = self.metrics[self.MACHINE_LEARNING]
        for metric_name in self.eval_dict[self.MACHINE_LEARNING]:
            metric_fun = self.criteria.get(metric_name)
            metric_dict[metric_name] = {}
            for nm_, pred_ in self.predictions.items():
                if nm_ == "__prod_p_dc":
                    # fix for the DC approximation
                    continue
                true_ = self.observations[nm_]
                tmp = metric_fun(true_, pred_)
                if isinstance(tmp, Iterable):
                    metric_dict[metric_name][nm_] = [float(el) for el in tmp]
                    self.logger.info("%s for %s: %s", metric_name, nm_, tmp)
                else:
                    metric_dict[metric_name][nm_] = float(tmp)
                    self.logger.info("%s for %s: %.2f", metric_name, nm_, tmp)

    def evaluate_physics(self):
        """
        function that evaluates the physics compliances on given observations
        It comprises various verifications which are:
            - Basic verifications (current/voltage/loss positivity, current eq, disc_lines)
            - Verification of law of conservation of energy
            - Verification of electrical loss
            - Verification of Kirchhoff's current law
            - Verification of Joule's law
        """
        # an example of how to call the physics compliances
        # physics_compliances.basic_verifier()
        metric_dict = self.metrics[self.PHYSICS_COMPLIANCES]
        for metric_name in self.eval_dict[self.PHYSICS_COMPLIANCES]:
            metric_fun = self.criteria.get(metric_name)
            metric_dict[metric_name] = {}
            tmp = metric_fun(self.predictions,
                             log_path=self.log_path,
                             observations=self.observations,
                             config=self.config)
            metric_dict[metric_name] = tmp

    def evaluate_industrial_readiness(self):
        """
        Evaluate the augmented simulators from Industrial Readiness point of view

        - Inference time
        - Scalability

        """
        metric_dict = self.metrics[self.INDUSTRIAL_READINESS]

    def evaluate_ood(self):
        """
        Evaluate the augmented simulators from Out-Of-Distribution Generalization point of view

        It considers a test dataset differently distributed than the learning dataset

        It operates on dataset with name `test_ood_topo_dataset`
        """
        metric_dict = self.metrics[self.OOD_GENERALIZATION]

