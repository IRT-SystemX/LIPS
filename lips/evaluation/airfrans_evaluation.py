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
from scipy.stats import spearmanr

from lips.config.configmanager import ConfigManager
from lips.logger import CustomLogger
from lips.evaluation import Evaluation
from lips.evaluation.utils import metric_factory
from lips.metrics.ml_metrics import metrics
from lips.dataset.scaler.standard_scaler_iterative import iterative_fit

import airfrans as af

def normalize_data(data,mean,std,field_names):
    flattened_data = np.concatenate([data[field_name][:, None] for field_name in field_names], axis = 1)
    flattened_data -= mean
    flattened_data /= std
    normalized_data = {field_name:flattened_data[:,field_id] for field_id,field_name in enumerate(field_names)}
    return normalized_data


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
        self.ml_normalization = None

    def evaluate(self,
                 observations: dict,
                 predictions: dict,
                 observation_metadata : dict,
                 ml_normalization : Union[dict, None]=None,
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
        criteria = {}
        self.ml_normalization = ml_normalization
        self.observation_metadata = observation_metadata

        for cat in self.eval_dict.keys():
            criteria = self._dispatch_evaluation(cat, criteria)

        return criteria

    def _dispatch_evaluation(self, category: str, criteria: dict):
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
                criteria[self.MACHINE_LEARNING] = self.evaluate_ml()
        if category == self.PHYSICS_COMPLIANCES:
            if self.eval_dict[category]:
                criteria[self.PHYSICS_COMPLIANCES] = self.evaluate_physics()
        if category == self.INDUSTRIAL_READINESS:
            raise Exception("Not done yet, sorry")
        
        return criteria

    def evaluate_ml(self) -> dict:
        """
        Verify AirfRANS Machine Learning metrics
        """
        self.logger.info("Evaluate machine learning metrics")
        #Normalize prediction and observation before computing mlMetrics

        field_names = self.predictions.keys()
        if self.ml_normalization is None:
            self.logger.info("Using current dataset to normalize")
            chunk_sizes = [int(simulation[1]) for simulation in self.observation_metadata["simulation_names"]]
            flattened_observation = np.concatenate([self.observations[field_name][:, None] for field_name in field_names], axis = 1)
            mean_observ,std_observ = iterative_fit(flattened_observation,chunk_sizes)
        else:
            self.logger.info("Using reference dataset data to normalize")
            mean_observ,std_observ = self.ml_normalization["mean"],self.ml_normalization["std"]
        normalized_predictions = normalize_data(data=self.predictions,mean=mean_observ,std=std_observ,field_names=field_names)
        normalized_observations = normalize_data(data=self.observations,mean=mean_observ,std=std_observ,field_names=field_names)


        metrics_ml = {}
        for metric_name in self.eval_dict[self.MACHINE_LEARNING]:
            metric_fun = metric_factory.get_metric(metric_name)
            metric_name_normalized = metric_name+"_normalized"
            metrics_ml[metric_name_normalized] = {}
            for nm_, pred_ in normalized_predictions.items():
                self.logger.info("Evaluating metric %s on variable %s", metric_name_normalized, nm_)
                true_ = normalized_observations[nm_]
                tmp = metric_fun(true_, pred_)

                if isinstance(tmp, Iterable):
                    metrics_ml[metric_name_normalized][nm_] = [float(el) for el in tmp]
                    self.logger.info("%s for %s: %s",metric_name_normalized, nm_, tmp)
                else:
                    metrics_ml[metric_name_normalized][nm_] = float(tmp)
                    self.logger.info("%s for %s: %.2E", metric_name_normalized, nm_, tmp)
            #Compute additional metric for pressure at surface
            true_pressure = normalized_observations["pressure"]
            pred_pressure = normalized_predictions["pressure"]
            surface_data=self.observation_metadata["surface"]
            tmp_surface = metric_fun(true_pressure[surface_data.astype(bool)], pred_pressure[surface_data.astype(bool)])
            metrics_ml[metric_name_normalized+"_surfacic"]={"pressure": float(tmp_surface)}
            self.logger.info("%s surfacic for %s: %s", metric_name_normalized, "pressure", tmp_surface)
        return metrics_ml
    
    def evaluate_physics(self) -> dict:
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

        return metrics_values

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
    attr_y = attr_names[7:-1]
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
    config_path_benchmark=get_root_path()+os.path.join("..","configurations","airfoil","benchmarks","confAirfoil.ini")
    evaluation = AirfRANSEvaluation(config_path=config_path_benchmark,scenario="Case1",data_path = directory_name, log_path = 'log_eval')
    output_values = {key:value for key,value in my_dataset.data.items() if key in attr_y}
    metrics = evaluation.evaluate(observations = output_values, predictions = output_values, observation_metadata = my_dataset.extra_data)
    print("Evaluation for solution equal to reference")
    print(metrics)
