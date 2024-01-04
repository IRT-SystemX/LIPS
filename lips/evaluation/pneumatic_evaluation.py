"""
Usage:
    Grid2opSimulator implementing powergrid physical simulator
Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""
from typing import Union
from collections.abc import Iterable
import numpy as np

from lips.config.configmanager import ConfigManager
from lips.physical_simulator.GetfemSimulator.GetfemSimulatorBridge import PhysicalCriteriaComputation
from lips.physical_simulator.getfemSimulator import GetfemSimulator
from lips.evaluation import Evaluation
from lips.logger import CustomLogger
from lips.evaluation.utils import metric_factory

class PneumaticEvaluation(Evaluation):
    """Evaluation of the pneumatic specific metrics

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
        super(PneumaticEvaluation,self).__init__(config=config,
                                                 config_path=config_path,
                                                 config_section=scenario,
                                                 log_path=log_path
                                                 )

        self.eval_dict = self.config.get_option("eval_dict")
        self.eval_params = self.config.get_option("eval_params")
        self.eval_crit_args = self.config.get_option("eval_crit_args")

        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        scenario_params=self.config.get_option("env_params")
        self.simulator = GetfemSimulator(**scenario_params)
        self.simulator.build_model()

    @classmethod
    def from_benchmark(cls,
                       benchmark: "WheelBenchmark",
                      ):
        """ Intialize the evaluation class from a benchmark object

        Parameters
        ----------
        benchmark
            a benchmark object

        Returns
        -------
        PneumaticEvaluation
        """
        return cls(config=benchmark.config, log_path=benchmark.log_path)

    def evaluate(self,
                 observations: dict,
                 predictions: dict,
                 save_path: Union[str, None]=None) -> dict:
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
        criteria = {}

        for cat in self.eval_dict.keys():
            self._dispatch_evaluation(cat, criteria)

        # TODO: save the self.metrics variable
        if save_path:
            pass

        return criteria

    def _dispatch_evaluation(self, category: str, criteria: dict):
        """
        This helper function select the evaluation function with respect to the category

        In PowerGrid case, the OOD generalization evaluation is performed using `Benchmark` class
        by iterating over all the datasets

        Parameters
        ----------
        category: `str`
            the evaluation criteria category, the values could be one of the [`ML`, `Physics`]
        """
        if category == self.MACHINE_LEARNING:
            if self.eval_dict[category]:
                criteria[self.MACHINE_LEARNING] = self.evaluate_ml()
        if category == self.PHYSICS_COMPLIANCES:
            if self.eval_dict[category]:
                criteria[self.PHYSICS_COMPLIANCES] = self.evaluate_physics()
        if category == self.INDUSTRIAL_READINESS:
            raise Exception("Not done yet, sorry")

        return criteria

    def evaluate_ml(self) -> dict:
        """
        Verify Pneumatic Machine Learning metrics
        """
        metrics_ml = {}
        #metric_val_by_name = self.metrics[self.MACHINE_LEARNING]
        for metric_name in self.eval_dict[self.MACHINE_LEARNING]:
            metric_fun = metric_factory.get_metric(metric_name)
            metrics_ml[metric_name] = {}
            for nm_, pred_ in self.predictions.items():
                true_ = self.observations[nm_]
                tmp = metric_fun(true_, pred_)
                if isinstance(tmp, Iterable):
                    metrics_ml[metric_name][nm_] = [float(el) for el in tmp]
                    self.logger.info("%s for %s: %s", metric_name, nm_, tmp)
                else:
                    metrics_ml[metric_name][nm_] = float(tmp)
                    self.logger.info("%s for %s: %.2E", metric_name, nm_, tmp)
        return metrics_ml

    def evaluate_physics(self) -> dict:
        """
        function that evaluates physical criteria on given observations and may rely on the physical solver
        """
        #metric_val_by_name = self.metrics[self.PHYSICS_COMPLIANCES]
        metrics_physics = {}
        attr_x=self.config.get_option("attr_x")
        obs_inputs={key: self.observations[key] for key in attr_x}
        inputs_separated = [dict(zip(obs_inputs,t)) for t in zip(*obs_inputs.values())]

        attr_y=self.config.get_option("attr_y_eval")
        obs_outputs={key: self.observations[key] for key in attr_y}
        output_separated = [dict(zip(obs_outputs,t)) for t in zip(*obs_outputs.values())]
        
        prediction_separated = [dict(zip(self.predictions,t)) for t in zip(*self.predictions.values())]

        metrics_physics = {metric_name:[] for metric_name in self.eval_dict[self.PHYSICS_COMPLIANCES]}
        for obs_input,obs_output,predict_out in zip(inputs_separated,output_separated,prediction_separated):
            simulator=type(self.simulator)(simulator_instance=self.simulator)
            simulator.modify_state(state=obs_input)
            simulator.build_model()

            for metric_name in self.eval_dict[self.PHYSICS_COMPLIANCES]:
                if self.eval_crit_args and self.eval_crit_args[metric_name]:
                    criteria_params=self.eval_crit_args[metric_name]
                else:
                    criteria_params=None
                obs_crit = PhysicalCriteriaComputation(criteria_type=metric_name,simulator=simulator,field=obs_output,criteria_params=criteria_params)
                pred_crit = PhysicalCriteriaComputation(criteria_type=metric_name,simulator=simulator,field=predict_out,criteria_params=criteria_params)

                delta_absolute=np.array(obs_crit)-np.array(pred_crit)
                delta_relative=delta_absolute/np.array(obs_crit)
                delta={
                    "absolute":delta_absolute,
                    "relative":delta_relative
                }
                metrics_physics[metric_name].append(delta)

        for metric_name in self.eval_dict[self.PHYSICS_COMPLIANCES]:
            deltas=metrics_physics[metric_name]
            deltas_united = {error_type: np.array([single_comparison[error_type] for single_comparison in deltas]) for error_type in deltas[0]}

            mean_relative_error=np.mean(deltas_united["relative"],axis=0)
            if isinstance(mean_relative_error, Iterable):
                for component_id,value in enumerate(mean_relative_error):
                    self.logger.info("%s mean relative error for component %d: %.2E", metric_name, component_id, value)
            else:
                self.logger.info("%s mean relative error for %.2E", metric_name, mean_relative_error)

            mean_absolute_error=np.mean(deltas_united["absolute"],axis=0)
            if isinstance(mean_absolute_error, Iterable):
                for component_id,value in enumerate(mean_absolute_error):
                    self.logger.info("%s mean absolute error for component %d: %.2E", metric_name, component_id, value)
            else:
                self.logger.info("%s mean absolute error for %.2E", metric_name, mean_absolute_error)

        return metrics_physics