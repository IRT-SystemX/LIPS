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
from lips.logger import CustomLogger

from lips.evaluation import Evaluation
from lips.evaluation.utils import metric_factory
from lips.metrics.ml_metrics import metrics

import airfrans as af
from scipy.stats import spearmanr

class AirfRANSEvaluation(Evaluation):
    """Evaluation of the AirfRANS specific metrics

    It is a subclass of the Evaluation class

    Parameters
    ----------
    config, optional
        an object of `ConfigManager` class
    config_path, optional
        _description_, by default None
    scenario, optional
        one of the airfRANS Scenario names, by default None
    log_path, optional
        path where the log should be maintained, by default None
    """
    def __init__(self,
                 data_path: str,
                 config: Union[ConfigManager, None]=None,
                 config_path: Union[str, None]=None,
                 scenario: Union[str, None]=None,
                 log_path: Union[str, None]=None
                 ):
        super(AirfRANSEvaluation,self).__init__(config=config,
                                                 config_path=config_path,
                                                 config_section=scenario,
                                                 log_path=log_path
                                                 )

        self.data_path = data_path
        self.eval_dict = self.config.get_option("eval_dict")
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        self.observation_metadata = dict()

    def evaluate(self,
                 observations: dict,
                 predictions: dict,
                 observation_metadata : dict,
                 save_path: Union[str, None]=None) -> dict:
        """The main function which evaluates all the required criteria noted in config file

        Parameters
        ----------
        observations
            observations from true observations used to evaluate the predictions
        predictions
            predictions obtained from augmented simulators
        save_path, optional
            path where the results should be saved, by default None
        """
        # call the base class for generic evaluations
        super().evaluate(observations, predictions, save_path)
        self.observation_metadata = observation_metadata

        for cat in self.eval_dict.keys():
            self._dispatch_evaluation(cat)

        return self.metrics

    def _dispatch_evaluation(self, category: str):
        """
        This helper function select the evaluation function with respect to the category

        In AirfRANS case, the OOD generalization evaluation is performed using `Benchmark` class
        by iterating over all the datasets

        Parameters
        ----------
        category: `str`
            the evaluation criteria category, the values could be one of the [`ML`, `Physics`]
        """
        if category == self.MACHINE_LEARNING:
            if self.eval_dict[category]:
                self.evaluate_ml()
        if category == self.PHYSICS_COMPLIANCES:
            if self.eval_dict[category]:
                self.evaluate_physics()
        if category == self.INDUSTRIAL_READINESS:
            raise Exception("Not done yet, sorry")

    def evaluate_ml(self):
        """
        Verify AirfRANS Machine Learning metrics
        """
        self.logger.info("Evaluate machine learning metrics")
        metric_val_by_name = self.metrics[self.MACHINE_LEARNING]
        self.metrics[self.MACHINE_LEARNING]={}
        for metric_name in self.eval_dict[self.MACHINE_LEARNING]:
            metric_fun = metric_factory.get_metric(metric_name)
            metric_val_by_name[metric_name] = {}
            for nm_, pred_ in self.predictions.items():
                self.logger.info("Evaluating metric %s on variable %s", metric_name, nm_)
                true_ = self.observations[nm_]
                tmp = metric_fun(true_, pred_)

                if isinstance(tmp, Iterable):
                    metric_val_by_name[metric_name][nm_] = [float(el) for el in tmp]
                    self.logger.info("%s for %s: %s", metric_name, nm_, tmp)
                else:
                    metric_val_by_name[metric_name][nm_] = float(tmp)
                    self.logger.info("%s for %s: %.2E", metric_name, nm_, tmp)
            self.metrics[self.MACHINE_LEARNING][metric_name] = metric_val_by_name[metric_name]

            #Compute additional metric for pressure at surface
            true_pressure = self.observations["pressure"]
            pred_pressure = self.predictions["pressure"]
            surface_data=self.observation_metadata["surface"]
            tmp_surface = metric_fun(true_pressure[surface_data.astype(bool)], pred_pressure[surface_data.astype(bool)])
            self.metrics[self.MACHINE_LEARNING][metric_name+"_surfacic"]={"pressure": float(tmp)}
            self.logger.info("%s surfacic for %s: %s", metric_name, "pressure", tmp_surface)

    def evaluate_physics(self):
        """
        Evaluate physical criteria on given observations
        """
        self.logger.info("Evaluate physical metrics")
        simulation_names=self.observation_metadata["simulation_names"]
        pred_data = self.from_batch_to_simulation(data=self.predictions,simulation_names=simulation_names)
        true_coefs = []
        coefs = []
        rel_err = []

        for n, simulation_name in enumerate(pred_data['simulation_names']):
            simulation = af.Simulation(root = self.data_path, name = simulation_name)
            simulation.velocity = np.concatenate([pred_data['x-velocity'][n][:, None], pred_data['y-velocity'][n][:, None]], axis = 1)
            simulation.pressure = pred_data['pressure'][n]
            simulation.nu_t = pred_data['turbulent_viscosity'][n]
            coefs.append(simulation.force_coefficient())
            rel_err.append(simulation.coefficient_relative_error())
            true_coefs.append(simulation.force_coefficient(reference = True))
        rel_err = np.array(rel_err)
        
        spear_drag = np.array([coefs[n][0][0] for n in range(len(coefs))])
        spear_true_drag = np.array([true_coefs[n][0][0] for n in range(len(true_coefs))])
        spear_lift = np.array([coefs[n][1][0] for n in range(len(coefs))])
        spear_true_lift = np.array([true_coefs[n][1][0] for n in range(len(true_coefs))])

        metrics_values = dict()
        spear_coefs = (spearmanr(spear_drag, spear_true_drag)[0], spearmanr(spear_lift, spear_true_lift)[0])
        metrics_values["spearman_correlation_drag"]=spear_coefs[0]
        self.logger.info('The spearman correlation for the drag coefficient is: {:.3f}'.format(spear_coefs[0]))
        metrics_values["spearman_correlation_lift"]=spear_coefs[1]
        self.logger.info('The spearman correlation for the lift coefficient is: {:.3f}'.format(spear_coefs[1]))

        mean_rel_err, std_rel_err = rel_err.mean(axis = 0), rel_err.std(axis = 0)
        metrics_values["mean_relative_drag"]=mean_rel_err[0]
        self.logger.info('The mean relative absolute error for the drag coefficient is: {:.3f}'.format(mean_rel_err[0]))
        metrics_values["std_relative_drag"]=std_rel_err[0]
        self.logger.info('The standard deviation of the relative absolute error for the drag coefficient is: {:.3f}'.format(std_rel_err[0]))
        metrics_values["mean_relative_lift"]=mean_rel_err[1]
        self.logger.info('The mean relative absolute error for the lift coefficient is: {:.3f}'.format(mean_rel_err[1]))
        metrics_values["std_relative_lift"]=std_rel_err[1]
        self.logger.info('The standard deviation of the relative absolute error for the lift coefficient is: {:.3f}'.format(std_rel_err[1]))

        self.metrics[self.PHYSICS_COMPLIANCES]=metrics_values
        return {'target_coefficients': true_coefs, 'predicted_coefficients': coefs, 'relative absolute error': rel_err}

    def from_batch_to_simulation(self, data, simulation_names):
        sim_data = {}
        keys = list(set(data.keys()) - set(['simulation_names']))
        for key in keys:
            sim_data[key] = []
            ind = 0
            for n in range(simulation_names.shape[0]):           
                sim_data[key].append(data[key][ind:(ind + int(simulation_names[n, 1]))])
                ind += int(simulation_names[n, 1])
        sim_data['simulation_names'] = simulation_names[:, 0]

        return sim_data

if __name__ == '__main__':
    import os

    from lips import get_root_path
    from lips.dataset.airfransDataSet import AirfRANSDataSet,download_data

    directory_name='Dataset'
    if not os.path.isdir("Dataset"):
         download_data(root_path=".", directory_name=directory_name)

    attr_names = (
        'x-position',
        'y-position',
        'x-inlet_velocity', 
        'y-inlet_velocity', 
        'distance_function', 
        'x-normals', 
        'y-normals', 
        'x-velocity', 
        'y-velocity', 
        'pressure', 
        'turbulent_viscosity',
        'surface'
    )
    attr_x = attr_names[:7]
    attr_y = attr_names[7:]
    my_dataset = AirfRANSDataSet(config = None,
                                 name = 'train',
                                 task = 'scarce',
                                 split = "training",
                                 attr_names = attr_names,
                                 log_path = 'log',
                                 attr_x = attr_x,
                                 attr_y = attr_y)
    my_dataset.load(path = directory_name)
    print(my_dataset)
    config_path_benchmark=get_root_path()+os.path.join("..","configurations","airfrans","benchmarks","confAirfoil.ini")
    evaluation = AirfRANSEvaluation(config_path=config_path_benchmark,scenario="Case1",data_path = directory_name, log_path = 'log_eval')
    evaluation.evaluate(observations = my_dataset.data, predictions = my_dataset.data)