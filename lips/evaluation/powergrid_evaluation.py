# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union
from collections.abc import Iterable
import time


from .evaluation import Evaluation
from .utils import metric_factory
from ..logger import CustomLogger
from ..config import ConfigManager
from ..physical_simulator.dcApproximationAS import DCApproximationAS
from ..metrics.power_grid import physics_compliances
from ..metrics.ml_metrics import metrics
from ..metrics.ml_metrics import external_metrics

class PowerGridEvaluation(Evaluation):
    """Evaluation of the power grid specific metrics

    It is a subclass of the Evaluation class

    Parameters
    ----------
    config, optional
        an object of `ConfigManager` class
    config_path, optional
        _description_, by default None
    scenario, optional
        one of the Power Grid Scenario names, by default None
    log_path, optional
        path where the log should be maintained, by default None
    """
    def __init__(self,
                 config: Union[ConfigManager, None]=None,
                 config_path: Union[str, None]=None,
                 scenario: Union[str, None]=None,
                 log_path: Union[str, None]=None
                 ):
        super().__init__(config=config,
                         config_path=config_path,
                         config_section=scenario,
                         log_path=log_path
                         )
        self.eval_dict = self.config.get_option("eval_dict")
        self.eval_params = self.config.get_option("eval_params")

        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        # read the criteria and their mapped functions for power grid
        # self.criteria = self.mapper.map_powergrid_criteria()

    @classmethod
    def from_benchmark(cls,
                       benchmark: "PowerGridBenchmark",
                      ):
        """ Intialize the evaluation class from a benchmark object

        Parameters
        ----------
        benchmark
            a benchmark object

        Returns
        -------
        PowerGridEvaluation
        """
        return cls(config=benchmark.config, log_path=benchmark.log_path)

    def evaluate(self,
                 observations: dict,
                 predictions: dict,
                 save_path: Union[str, None]=None,
                 **kwargs) -> dict:
        """The main function which evaluates all the required criteria noted in config file

        Parameters
        ----------
        dataset
            DataSet object including true observations used to evaluate the predictions
        predictions
            predictions obtained from augmented simulators
        save_path, optional
            path where the results should be saved, by default None
        """
        # call the base class for generic evaluations
        super().evaluate(observations, predictions, save_path)

        # evaluate powergrid specific evaluations based on config
        for cat in self.eval_dict.keys():
            self._dispatch_evaluation(cat, **kwargs)

        # TODO: save the self.metrics variable
        if save_path:
            pass

        return self.metrics

    def _dispatch_evaluation(self, category: str, **kwargs):
        """
        This helper function select the evaluation function with respect to the category

        In PowerGrid case, the OOD generalization evaluation is performed using `Benchmark` class
        by iterating over all the datasets

        Parameters
        ----------
        category: `str`
            the evaluation criteria category, the values could be one of the [`ML`, `Physics`, `IndRed`, `OOD`]
        """
        if category == self.MACHINE_LEARNING:
            if self.eval_dict[category]:
                self.evaluate_ml(**kwargs)
        if category == self.PHYSICS_COMPLIANCES:
            if self.eval_dict[category]:
                self.evaluate_physics(**kwargs)
        if category == self.INDUSTRIAL_READINESS:
            if self.eval_dict[category]:
                self.evaluate_industrial_readiness(**kwargs)

    def evaluate_ml(self, **kwargs):
        """
        Verify PowerGrid Specific Machine Learning metrics such as MAPE90
        """
        metric_dict = self.metrics[self.MACHINE_LEARNING]
        for metric_name in self.eval_dict[self.MACHINE_LEARNING]:
            if metric_name == "TIME_INF":
                try:
                    augmented_simulator = kwargs["augmented_simulator"]
                    dataset = kwargs["dataset"]
                    if not isinstance(augmented_simulator, DCApproximationAS):
                        # using the machine learning point of view (max possible batch_size)
                        beg_ = time.perf_counter()
                        _ = augmented_simulator.predict(dataset, eval_batch_size=dataset.size)
                        end_ = time.perf_counter()
                        total_time = end_ - beg_
                        metric_dict[metric_name] = total_time
                        self.logger.info("%s for %s: %s", metric_name, augmented_simulator.name, total_time)
                    else:
                        dc_comp_time = augmented_simulator.comp_time
                        metric_dict[metric_name] = dc_comp_time
                        self.logger.info("%s for %s: %s", metric_name, augmented_simulator.name, dc_comp_time)
                except KeyError:
                    self.logger.error("The augmented simulator or dataset are not provided to estimate the inference time.")
            else:
                metric_fun = metric_factory.get_metric(metric_name)
                # metric_fun = self.criteria.get(metric_name)
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

    def evaluate_physics(self, **kwargs):
        """
        function that evaluates the physics compliances on given observations
        It comprises various verifications which are:

        - Basic verifications (current/voltage/loss positivity, current eq, disc_lines)
        - Verification of law of conservation of energy
        - Verification of electrical loss
        - Verification of Kirchhoff's current law
        - Verification of Joule's law
        """
        try:
            env = kwargs["env"]
        except KeyError:
            self.logger.error("The environment (physical solver) is required for some physics critiera and should be provided")
        metric_dict = self.metrics[self.PHYSICS_COMPLIANCES]
        for metric_name in self.eval_dict[self.PHYSICS_COMPLIANCES]:
            metric_fun = metric_factory.get_metric(metric_name)
            #metric_fun = self.criteria.get(metric_name)
            metric_dict[metric_name] = {}
            tmp = metric_fun(self.predictions,
                             log_path=self.log_path,
                             observations=self.observations,
                             config=self.config,
                             **kwargs)
                             #env=env)
            metric_dict[metric_name] = tmp

    def evaluate_industrial_readiness(self, **kwargs):
        """
        Evaluate the augmented simulators from Industrial Readiness point of view

        - Inference time
        - Scalability

        """
        metric_dict = self.metrics[self.INDUSTRIAL_READINESS]
        for metric_name in self.eval_dict[self.INDUSTRIAL_READINESS]:
            if metric_name == "TIME_INF":
                try:
                    augmented_simulator = kwargs["augmented_simulator"]
                    dataset = kwargs["dataset"]

                    if not isinstance(augmented_simulator, DCApproximationAS):
                        try:
                            eval_batch_size = self.eval_params["inf_batch_size"]
                        except KeyError:
                            eval_batch_size = kwargs.get("eval_batch_size", augmented_simulator.params["eval_batch_size"])

                        # using the industrial readiness point of view (max possible batch_size)
                        beg_ = time.perf_counter()
                        _ = augmented_simulator.predict(dataset, eval_batch_size=eval_batch_size)
                        end_ = time.perf_counter()
                        total_time = end_ - beg_
                        metric_dict[metric_name] = total_time
                        self.logger.info("%s for %s: %s", metric_name, augmented_simulator.name, total_time)

                    else:
                        dc_comp_time = augmented_simulator.comp_time
                        metric_dict[metric_name] = dc_comp_time
                        self.logger.info("%s for %s: %s", metric_name, augmented_simulator.name, dc_comp_time)
                except KeyError:
                    self.logger.error("The augmented simulator or dataset are not provided to estimate the inference time.")
