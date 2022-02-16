# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import json
import copy
from typing import Union
import numpy as np

from collections.abc import Iterable

from lips.logger import CustomLogger

from lips.metrics import DEFAULT_METRICS
from lips.metrics import metricPercentage
from lips.evaluation.BasicVerifier import BasicVerifier
from lips.evaluation.Check_loss import Check_loss
try:
    from lips.evaluation.Check_Kirchhoff_current_law import Check_Kirchhoff_current_law
    CHECK_KIRCHHOFF_AVAIL = True
except ImportError as exc_:
    # grid2op is not installed, but this should work as is
    CHECK_KIRCHHOFF_AVAIL = False

from lips.evaluation.Check_KCL import check_kcl

from lips.evaluation.Check_energy_conservation import Check_energy_conservation


ERROR_MSG_KCL = "Impossible to check for the Kirchhoff current laws "\
                "because some requirements are not met (probably a "\
                "missing dependency)"

#logging.basicConfig(filename="logs.log",
#                    level=logging.INFO,
#                    format="%(levelname)s:%(message)s")


class Evaluation(object):
    """
    this class can be used to evaluate the designed models from different aspects 

    Physics compliance : is the learned model respect the physics laws (at which degree)
    ML-related performance : how the model respect the ML related criteria like accuracy of predictions
    Readiness : Is the designed model ready to be deployed in production (scalibility or stability)
    Generalization : how much robustness and out-of-distribution generalization is necessary ?

    
    
    The verification are done on the basis of active_dict member variable which is a dictionary
        a dictionary indicating to activate or deactivate each of the verifications by indicating `True` or `False`
        for each verification. An empty dict can be retrieved by calling ``get_active_dict()``
        
        The keys should be :
            `evaluate_ML`
            `evaluate_physic`
                # here are nested keys under physics compliances
                `verify_current_pos` # verifies the positivity of currents
                `verify_voltage_pos` # verifies the positivity of voltages
                `verify_loss_pos` # verifies the positivity of electrical loss
                `verify_predict_disc` # verifies if the predicted flow is null for a corresponding disconnected line
                `verify_current_eq` # verifies the current equations for each extremity of a power line
                `verify_EL` # verifies the electrical loss
                `verify_LCE` # verifies the law of conservation of energy
                `verify_KCL` # verifies the Kirchhoff's current law
            `evaluate_adaptability`
            `evaluate_readiness`

    params
    ------
        benchmark: ``object`` of class ``Benchmark``
            it includes some model specific information such as training and prediction time

        observations : ``list``
            list of real observations used during test and evaluation

        predictions : ``list``
            a list of predictions of the model

        save_path : ``str``
            the path to save the evaluation results
        

    """

    def __init__(self,
                 log_path: Union[str, None]=None
                ):
        
        #self.benchmark = benchmark
        #self.observations = copy.deepcopy(self.benchmark.observations[tag])
        #self.predictions = copy.deepcopy(self.benchmark.predictions[tag])

        self.env = None
        self.env_name = None

        self.observations = None
        self.predictions = None

        self.active_dict = self.get_empty_active_dict()

        self.save_path = None 

        self.metrics_ML = {}
        self.metrics_physics = {}
        self.metrics_generalization = {}
        self.metrics_readiness = {}

        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        # create the paths
        # if self.benchmark.model_path save the evaluations in this directory with the corresponding tag
        # if self.benchmark.dc_path, save the evaluations in this directory with the corresponding tag

    def do_evaluations(self, 
                       env, 
                       env_name, 
                       observations, 
                       predictions,
                       choice="predictions",
                       EL_tolerance=0.04,
                       LCE_tolerance=1e-3,
                       KCL_tolerance=1e-2,
                       active_flow=True,
                       save_path=None):
        self.env = env
        self.env_name = env_name
        self.observations = copy.deepcopy(observations)
        self.predictions = copy.deepcopy(predictions)
        
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        
        if self.active_dict["evaluate_ML"]:
            self.evaluate_ML(save_path=save_path)

        self.evaluate_physic(choice=choice,
                             EL_tolerance=EL_tolerance,
                             LCE_tolerance=LCE_tolerance,
                             KCL_tolerance=KCL_tolerance,
                             active_flow=active_flow,
                             save_path=save_path)

        if self.active_dict["evaluate_adaptability"]:
            self.evaluate_generalization()

        if self.active_dict["evaluate_readiness"]:
            self.evaluate_readiness()

        return self.metrics_ML, self.metrics_physics, self.metrics_generalization, self.metrics_readiness

    def evaluate_physic(self,
                        choice="real",
                        EL_tolerance=0.04,
                        LCE_tolerance=1e-3,
                        KCL_tolerance=1e-2,
                        active_flow=True,
                        save_path=None):
        """
        function that evaluates the physics compliances on given observations
        It comprises various verifications which are : 
            1) Basic verifications
            2) Verification of law of conservation of energy 
            3) Verification of electrical loss
            4) Verification of Kirchhoff's current law

        TODO : implement the save and load of the evaluations

        params
        ------
            basic_verifier: ``bool``
                whether to check the basic physics 

            check_EL: ``bool``
                whether to ckeck the electrical loss values (EL)

            check_LCE: ``bool``
                whether to check the law of conservation of energy (LCE)

            check KCL: ``bool``
                whether to check the Kirchhoff's current law (KCL)

            choice: ``str``
                whether the evaluation is performed on real observations or predictions
                the values could be `real` or `predictions`

            EL_tolerance: ``float``
                the tolerance used for electrical loss verification

            LCE_tolerance: ``float``
                the tolerance used for the verification of law of conservation of energy

            KCL_tolerance: ``float``
                the tolerance used for the verification of Kirchhoff's current law 

        """
        ##################################
        ########### observations #########
        ##################################
        if choice == "real":
            ######### Basic Physics verification
            self.metrics_physics['BasicVerifications'] = {}
            verifications = BasicVerifier(predictions=self.observations,
                                          line_status=self.observations["line_status"],
                                          active_dict=self.active_dict["evaluate_physic"],
                                          log_path=self.log_path
                                          )
            self.metrics_physics['BasicVerifications'] = verifications

            ######### Electrical loss verifier
            if self.active_dict["evaluate_physic"]["verify_EL"]:
                self.metrics_physics['EL'] = {}
                loss_metrics = Check_loss(p_or=self.observations["p_or"],
                                          p_ex=self.observations["p_ex"],
                                          prod_p=self.observations["prod_p"],
                                          tolerance=EL_tolerance,
                                          log_path=self.log_path
                                          )
                self.metrics_physics['EL']['violation_percentage'] = float(loss_metrics[1])
                self.metrics_physics['EL']['EL_values'] = [float(el) for el in np.array(loss_metrics[0])]

            ######### Law of conservation of energy verifier
            if self.active_dict["evaluate_physic"]["verify_LCE"]:
                self.metrics_physics['LCE'] = {}
                lce_metrics = Check_energy_conservation(prod_p=self.observations["prod_p"],
                                                        load_p=self.observations["load_p"],
                                                        p_or=self.observations["p_or"],
                                                        p_ex=self.observations["p_ex"],
                                                        tolerance=LCE_tolerance,
                                                        log_path=self.log_path
                                                        )
                
                self.metrics_physics['LCE']['LCE_values'] = [float(el) for el in np.array(lce_metrics[0])]
                self.metrics_physics['LCE']['violation_percentage'] = float(lce_metrics[1])
                self.metrics_physics['LCE']['MAE'] = float(lce_metrics[3])

            ######### Kirchhoff's current law verifier
            if self.active_dict["evaluate_physic"]["verify_KCL"]:
                if not CHECK_KIRCHHOFF_AVAIL:
                    raise RuntimeError(ERROR_MSG_KCL)
                self.metrics_physics["KCL"] = {}
                res_kcl = Check_Kirchhoff_current_law(env=self.env,
                                                      env_name=self.env_name,
                                                      data=self.observations,
                                                      load_p=self.observations["load_p"],
                                                      load_q=self.observations["load_q"],
                                                      prod_p=self.observations["prod_p"],
                                                      prod_v=self.observations["prod_v"],
                                                      line_status=self.observations["line_status"],
                                                      topo_vect=self.observations["topo_vect"],
                                                      active_flow=active_flow,
                                                      tolerance=KCL_tolerance,
                                                      log_path=self.log_path
                                                      )
                self.metrics_physics["KCL"]["violation_percentage"] = float(res_kcl[3])
                self.metrics_physics["KCL"]["nodes_values"] = res_kcl[0]#[float(el) for el in res_kcl[0]]
                self.metrics_physics["KCL"]["network_values"] = [float(el) for el in np.array(res_kcl[1])]
                self.metrics_physics["KCL"]["violation_indices"] = [int(el) for el in np.array(res_kcl[2])]

        ##################################
        ########### predictions ##########
        ##################################
        elif choice == "predictions":
            ######### Basic physics verifications
            self.metrics_physics['BasicVerifications'] = {}
            verifications = BasicVerifier(predictions=self.predictions,
                                          line_status=self.observations["line_status"],
                                          active_dict=self.active_dict["evaluate_physic"],
                                          log_path=self.log_path
                                          )
            self.metrics_physics['BasicVerifications'] = verifications

            ######### Electrical loss verifier
            if self.active_dict["evaluate_physic"]["verify_EL"]:
                self.metrics_physics['EL'] = {}
                loss_metrics = Check_loss(p_or=self.predictions["p_or"],
                                          p_ex=self.predictions["p_ex"],
                                          prod_p=self.observations["prod_p"],
                                          tolerance=EL_tolerance,
                                          log_path=self.log_path
                                          )
                self.metrics_physics['EL']['violation_percentage'] = float(loss_metrics[1])
                self.metrics_physics['EL']['EL_values'] = [float(el) for el in np.array(loss_metrics[0])]

            ######### Law of conservation of energy verifier
            if self.active_dict["evaluate_physic"]["verify_LCE"]:
                self.metrics_physics['LCE'] = {}
                lce_metrics = Check_energy_conservation(prod_p=self.observations["prod_p"],
                                                        load_p=self.observations["load_p"],
                                                        p_or=self.predictions["p_or"],
                                                        p_ex=self.predictions["p_ex"],
                                                        tolerance=LCE_tolerance,
                                                        log_path=self.log_path
                                                        )
                
                self.metrics_physics['LCE']['LCE_values'] = [float(el) for el in np.array(lce_metrics[0])]
                self.metrics_physics['LCE']['violation_percentage'] = float(lce_metrics[1])
                self.metrics_physics['LCE']['MAE'] = float(lce_metrics[3])

            # Kirchhoff's current law verifier
            if self.active_dict["evaluate_physic"]["verify_KCL"]:
                if not CHECK_KIRCHHOFF_AVAIL:
                    raise RuntimeError(ERROR_MSG_KCL)
                self.metrics_physics["KCL"] = {}
                self.metrics_physics["KCL_new"] = {}
                res_kcl = Check_Kirchhoff_current_law(env=self.env,
                                                      env_name=self.env_name,
                                                      data=self.predictions,
                                                      load_p=self.observations["load_p"],
                                                      load_q=self.observations["load_q"],
                                                      prod_p=self.observations["prod_p"],
                                                      prod_v=self.observations["prod_v"],
                                                      line_status=self.observations["line_status"],
                                                      topo_vect=self.observations["topo_vect"],
                                                      active_flow=active_flow,
                                                      tolerance=KCL_tolerance,
                                                      log_path=self.log_path
                                                      )
                self.metrics_physics["KCL"]["violation_percentage"] = float(res_kcl[3])
                self.metrics_physics["KCL"]["nodes_values"] = res_kcl[0]
                self.metrics_physics["KCL"]["network_values"] = [float(el) for el in np.array(res_kcl[1])]
                self.metrics_physics["KCL"]["violation_indices"] = [int(el) for el in np.array(res_kcl[2])]

                res_kcl_new = check_kcl(env=self.env,
                                        ref_obs=self.observations,
                                        predictions=self.predictions,
                                        tol=KCL_tolerance
                                        )

                self.metrics_physics["KCL_new"]["violation_prop_obs_level"] = float(res_kcl_new[0])
                self.metrics_physics["KCL_new"]["violation_prop_sub_level"] = float(res_kcl_new[1])

        else:
            raise ValueError

        if save_path is not None:
            res = self._save_complex_dict(self.metrics_physics)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path, "metrics_physic.json"), "w", encoding="utf-8") as f:
                json.dump(res, fp=f, indent=4, sort_keys=True)
            self.save_path = save_path
        elif self.save_path is not None:
            res = self._save_complex_dict(self.metrics_physics)
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            with open(os.path.join(self.save_path, "metrics_physic.json"), "w", encoding="utf-8") as f:
                json.dump(res, fp=f, indent=4, sort_keys=True)
            self.save_path = save_path

        return self.metrics_physics

    def evaluate_ML(self,
                    observations: Union[dict, None]=None,
                    predictions: Union[dict, None]=None,
                    metric_names: Union[list, None]=None,
                    compute_metricsPercentage: bool=True,
                    metric_percentage: set=("mape", "MAE"),
                    k: float=0.1,
                    verbose: int=0,
                    save_path: Union[str, None]=None):
        """
        Machine learning evaluation metrics including inference time and prediction accuracy

        params
        ------
            metric_names : ``list`` of ``str``
                a list of metric names which should be computed between predictions and real observations
                TODO : add MAPE90 among the metrics

            compute_metricsPercentage: ``bool``
                whether or not to compute the metrics on a percentage of highest current values. 
                This is important for power network security (a metric as MAPE90 or MAE90)

            metric_percentage: ``list``
                a list of metrics used for evaluation on k% highest values of current for each line

            k: ``float``
                indicate the proportion of highest values to be considered in metrics computations

            verbose : ```bool``
                whether to print the results of computed metrics or not

            save: ``bool``
                whether to save the results


        """
        self.logger.info("Machine learning metrics")
        if observations: 
            self.observations = observations
        if predictions:
            self.predictions = predictions
        if metric_names is None:
            metrics = DEFAULT_METRICS
        else:
            metrics = {nm_: DEFAULT_METRICS[nm_] for nm_ in metric_names}

        #self.metrics_ML["train_time"] = self.benchmark.training_time
        #self.metrics_ML["test_time"] = self.benchmark.prediction_time

        for metric_name, metric_fun in metrics.items():
            self.metrics_ML[metric_name] = {}
            for nm, pred_ in self.predictions.items():
                if nm == "__prod_p_dc":
                    # fix for the DC approximation
                    continue
                true_ = self.observations[nm]
                tmp = metric_fun(true_, pred_)
                if isinstance(tmp, Iterable):
                    if verbose >= 2:
                        print(f"{metric_name} for {nm}: {tmp}")
                    self.metrics_ML[metric_name][nm] = [
                        float(el) for el in tmp]
                else:
                    if verbose >= 1:
                        print(f"{metric_name} for {nm}: {tmp:.2f}")
                    self.metrics_ML[metric_name][nm] = float(tmp)

        if compute_metricsPercentage:
            metricPercentage(self.metrics_ML,
                             observations=self.observations,
                             predictions=self.predictions,
                             k=k,
                             metric_names=metric_percentage,
                             variables=self.observations.keys(),
                             agg_func=np.mean,
                             log_path=self.log_path
                             )

        # save the results in a json file
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path, "metrics_ML.json"), "w", encoding="utf-8") as f:
                json.dump(self.metrics_ML, fp=f, indent=4, sort_keys=True)
            self.save_path = save_path
        elif self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            with open(os.path.join(self.save_path, "metrics_ML.json"), "w", encoding="utf-8") as f:
                json.dump(self.metrics_ML, fp=f, indent=4, sort_keys=True)

        return self.metrics_ML

    def evaluate_readiness(self):
        pass

    def evaluate_generalization(self):
        pass

    @staticmethod    
    def get_empty_active_dict():
        """
        it returns an empty active_dict to be parameterized in function of requirement and model outputs
        
        The keys of the dictionary are : 
        ---------------
        `evaluate_ML`
        `evaluate_physic`
            # here are nested keys under physics compliances
            `verify_current_pos` # verifies the positivity of currents
            `verify_voltage_pos` # verifies the positivity of voltages
            `verify_loss_pos` # verifies the positivity of electrical loss
            `verify_predict_disc` # verifies if the predicted flow is null for a corresponding disconnected line
            `verify_current_eq` # verifies the current equations for each extremity of a power line
            `verify_EL` # verifies the electrical loss
            `verify_LCE` # verifies the law of conservation of energy
            `verify_KCL` # verifies the Kirchhoff's current law
        `evaluate_adaptability`
        `evaluate_readiness`

        Returns
        --------
            active_dict: ``dict``
                a dictionary indicating to verify or not the evaluation criteria
        """
        active_dict = dict()
        active_dict["evaluate_ML"] = False
        active_dict["evaluate_adaptability"] = False
        active_dict["evaluate_readiness"] = False
        active_dict["evaluate_physic"] = dict()
        active_dict["evaluate_physic"]["verify_current_pos"] = False
        active_dict["evaluate_physic"]["verify_voltage_pos"] = False
        active_dict["evaluate_physic"]["verify_loss_pos"] = False
        active_dict["evaluate_physic"]["verify_predict_disc"] = False
        active_dict["evaluate_physic"]["verify_current_eq"] = False
        active_dict["evaluate_physic"]["verify_EL"] = False
        active_dict["evaluate_physic"]["verify_LCE"] = False
        active_dict["evaluate_physic"]["verify_KCL"] = False
        return active_dict

    def get_active_dict(self):
        return self.active_dict

    def set_active_dict(self, dict):
        """
        set the active_dict from user input
        """
        self.active_dict = dict

    def activate_physic_compliances(self):
        """
        active all the physics compliances verifications
        """
        self.active_dict["evaluate_physic"]["verify_current_pos"] = True
        self.active_dict["evaluate_physic"]["verify_voltage_pos"] = True
        self.active_dict["evaluate_physic"]["verify_loss_pos"] = True
        self.active_dict["evaluate_physic"]["verify_predict_disc"] = True
        self.active_dict["evaluate_physic"]["verify_current_eq"] = True
        self.active_dict["evaluate_physic"]["verify_EL"] = True
        self.active_dict["evaluate_physic"]["verify_LCE"] = True
        self.active_dict["evaluate_physic"]["verify_KCL"] = True

    def visualize_network_state(self):
        """
        visualize the network state for each observation during the evaluation phase
        """
        pass

    def plot_error(self):
        """
        plot any error in a plot
        """
        pass

    def _save_complex_dict(self, metric):
        res = dict()
        for key_, val_ in metric.items():
            res[key_] = dict()
            if isinstance(val_, dict):
                for key__, val__ in val_.items():
                    if isinstance(val__, Iterable):
                        res[key_][key__] = []
                        for el in val__:
                            self._save_dict(res[key_][key__], el)
                    else:
                        res[key_][key__] = np.float(val__)
                            
            else:
                res[key_] = []
                for el in val_:
                    self._save_dict(res[key_], el)
        return res

    def _save_dict(self, li, val):
        if isinstance(val, Iterable):
            li.append([float(el) for el in val])
        else:
            li.append(float(val))

    def load_results(self, load_path, metrics=["metrics_physics"]):
        """
        load the metrics
        """
        for metric in metrics:
            with open(os.path.join(load_path, f"{metric}.json"), "r", encoding="utf-8") as f:
                dict_serialized = json.load(fp=f) 
            
            for key_, val_ in dict_serialized.items():
                if isinstance(val_, dict):
                    for key__, val__ in val_.items():
                        if isinstance(val__, Iterable):
                            dict_serialized[key_][key__] = np.array(val__)
                        else:
                            dict_serialized[key_][key__] = np.float(val__)
                else:
                    if isinstance(val_, Iterable):
                        dict_serialized[key_] = np.array(val_)
                    else:
                        dict_serialized[key_] = np.float(val_)

            if metric == "metrics_physics":
                self.metrics_physics = dict_serialized
            elif metric == "metrics_ML":
                self.metrics_ML = dict_serialized
            elif metric == "metrics_readiness":
                self.metrics_generalization = dict_serialized
            elif metric == "metrics_generalization":
                self.metrics_readiness = dict_serialized
            else:
                raise NotImplementedError
