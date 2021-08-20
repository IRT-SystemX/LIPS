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

import numpy as np

from collections.abc import Iterable

from lips.metrics import DEFAULT_METRICS
from lips.metrics import metricPercentage
#from lips.evaluation import BasicVerifier, Check_loss, Check_Kirchhoff_current_law, Check_energy_conservation
from lips.evaluation.BasicVerifier import BasicVerifier
from lips.evaluation.Check_loss import Check_loss
from lips.evaluation.Check_Kirchhoff_current_law import Check_Kirchhoff_current_law
from lips.evaluation.Check_energy_conservation import Check_energy_conservation


class Evaluation():
    """
    this class can be used to evaluate the designed models from different aspects 

    Physics compliance : is the learned model respect the physics laws (at which degree)
    ML-related performance : how the model respect the ML related criteria like accuracy of predictions
    Readiness : Is the designed model ready to be deployed in production (scalibility or stability)
    Adaptability : how much robustness and out-of-distribution generalization is necessary ?

    params
    ------
        model : tensorflow ``Model``
            the trained model in Benchmark class

        observations : ``list``
            list of real observations used during test and evaluation

        predictions : ``list``
            a list of predictions of the model


    """

    def __init__(self, benchmark, tag):
        self.benchmark = benchmark
        self.observations = copy.deepcopy(self.benchmark.observations[tag])
        self.predictions = copy.deepcopy(self.benchmark.predictions[tag])

        self.metrics_ML = {}
        self.metrics_physics = {}
        self.metrics_adaptability = {}
        self.metrics_readiness = {}

        # create the paths
        # if self.benchmark.model_path save the evaluations in this directory with the corresponding tag
        # if self.benchmark.dc_path, save the evaluations in this directory with the corresponding tag

    def evaluate_physic(self,
                        basic_verifier=True,
                        check_EL=True,
                        check_LCE=True,
                        check_KCL=True,
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
            if basic_verifier:
                self.metrics_physics['BasicVerifications'] = {}
                verifications = BasicVerifier(a_or=self.observations["a_or"],
                                              a_ex=self.observations["a_ex"],
                                              v_or=self.observations["v_or"],
                                              v_ex=self.observations["v_ex"],
                                              p_or=self.observations["p_or"],
                                              p_ex=self.observations["p_ex"],
                                              q_or=self.observations["q_or"],
                                              q_ex=self.observations["q_ex"],
                                              line_status=self.observations["line_status"])
                self.metrics_physics['BasicVerifications'] = verifications

            if check_EL:
                self.metrics_physics['EL'] = {}
                loss_metrics = Check_loss(p_or=self.observations["p_or"],
                                          p_ex=self.observations["p_ex"],
                                          prod_p=self.observations["prod_p"],
                                          tolerance=EL_tolerance
                                          )
                self.metrics_physics['EL']['violation_percentage'] = loss_metrics[1]
                self.metrics_physics['EL']['EL_values'] = loss_metrics[0]

            if check_LCE:
                self.metrics_physics['LCE'] = {}
                lce_metrics = Check_energy_conservation(prod_p=self.observations["prod_p"],
                                                        load_p=self.observations["load_p"],
                                                        p_or=self.observations["p_or"],
                                                        p_ex=self.observations["p_ex"],
                                                        tolerance=LCE_tolerance)
                self.metrics_physics['LCE']['violation_percentage'] = lce_metrics[1]
                self.metrics_physics['LCE']['LCE_values'] = lce_metrics[0]

            if check_KCL:
                self.metrics_physics["KCL"] = {}
                res_kcl = Check_Kirchhoff_current_law(env=self.benchmark.dataset.env,
                                                      env_name=self.benchmark.dataset.env_name,
                                                      data=self.observations,
                                                      load_p=self.observations["load_p"],
                                                      load_q=self.observations["load_q"],
                                                      prod_p=self.observations["prod_p"],
                                                      prod_v=self.observations["prod_v"],
                                                      line_status=self.observations["line_status"],
                                                      topo_vect=self.observations["topo_vect"],
                                                      active_flow=active_flow,
                                                      tolerance=KCL_tolerance)
                self.metrics_physics["KCL"]["violation_percentage"] = res_kcl[3]
                self.metrics_physics["KCL"]["nodes_values"] = res_kcl[0]
                self.metrics_physics["KCL"]["network_values"] = res_kcl[1]

        ##################################
        ########### predictions ##########
        ##################################
        elif choice == "predictions":
            if basic_verifier:
                self.metrics_physics['BasicVerifications'] = {}
                verifications = BasicVerifier(a_or=self.predictions["a_or"],
                                              a_ex=self.predictions["a_ex"],
                                              v_or=self.predictions["v_or"],
                                              v_ex=self.predictions["v_ex"],
                                              p_or=self.predictions["p_or"],
                                              p_ex=self.predictions["p_ex"],
                                              q_or=self.predictions["q_or"],
                                              q_ex=self.predictions["q_ex"],
                                              line_status=self.observations["line_status"])
                self.metrics_physics['BasicVerifications'] = verifications

            if check_EL:
                self.metrics_physics['EL'] = {}
                loss_metrics = Check_loss(p_or=self.predictions["p_or"],
                                          p_ex=self.predictions["p_ex"],
                                          prod_p=self.observations["prod_p"],
                                          tolerance=EL_tolerance
                                          )
                self.metrics_physics['EL']['violation_percentage'] = loss_metrics[1]
                self.metrics_physics['EL']['EL_values'] = loss_metrics[0]

            if check_LCE:
                self.metrics_physics['LCE'] = {}
                lce_metrics = Check_energy_conservation(prod_p=self.observations["prod_p"],
                                                        load_p=self.observations["load_p"],
                                                        p_or=self.predictions["p_or"],
                                                        p_ex=self.predictions["p_ex"],
                                                        tolerance=LCE_tolerance)
                self.metrics_physics['LCE']['violation_percentage'] = lce_metrics[1]
                self.metrics_physics['LCE']['LCE_values'] = lce_metrics[0]

            if check_KCL:
                self.metrics_physics["KCL"] = {}
                res_kcl = Check_Kirchhoff_current_law(env=self.benchmark.dataset.env,
                                                      env_name=self.benchmark.dataset.env_name,
                                                      data=self.predictions,
                                                      load_p=self.observations["load_p"],
                                                      load_q=self.observations["load_q"],
                                                      prod_p=self.observations["prod_p"],
                                                      prod_v=self.observations["prod_v"],
                                                      line_status=self.observations["line_status"],
                                                      topo_vect=self.observations["topo_vect"],
                                                      active_flow=active_flow,
                                                      tolerance=KCL_tolerance)
                self.metrics_physics["KCL"]["violation_percentage"] = res_kcl[3]
                self.metrics_physics["KCL"]["nodes_values"] = res_kcl[0]
                self.metrics_physics["KCL"]["network_values"] = res_kcl[1]

        else:
            raise NotImplementedError

    def evaluate_ML(self,
                    metric_names=None,
                    save_path=None,
                    compute_metricsPercentage=True,
                    metric_percentage=["mape", "MAE"],
                    k=0.1,
                    verbose=0):
        """
        Machine learning evaluation metrics including inference time and prediction accuracy

        params
        ------
            metric_names : ``list`` of ``str``
                a list of metric names which should be computed between predictions and real observations
                TODO : add MAPE90 among the metrics

            save_path : ```str``
                if indicated, the evaluation results will be saved in the path

            compute_metricsPercentage: ``bool``
                whether or not to compute the metrics on a percentage of highest current values. 
                This is important for power network security (a metric as MAPE90 or MAE90)

            metric_percentage: ``list``
                a list of metrics used for evaluation of k% highest values of current

            k: ``float``
                indicate the proportion of highest values to be considered in metrics computations

            verbose : ```bool``
                whether to print the results of computed metrics or not


        """
        if metric_names is None:
            metrics = DEFAULT_METRICS
        else:
            metrics = {nm_: DEFAULT_METRICS[nm_] for nm_ in metric_names}

        self.metrics_ML["train_time"] = self.benchmark.training_time
        self.metrics_ML["test_time"] = self.benchmark.prediction_time

        for metric_name, metric_fun in metrics.items():
            self.metrics_ML[metric_name] = {}
            for nm, pred_ in self.predictions.items():
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
                             metric_names=metric_percentage,
                             observations=self.observations,
                             predictions=self.predictions,
                             k=k,
                             variables=["a_or", "p_or", "v_or", "q_or"])

        # save the results in a json file
        if save_path is not None:
            save_path = os.path.join(save_path, self.benchmark.benchmark_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(self.metrics_ML, fp=f, indent=4, sort_keys=True)

    def evaluate_readiness(self):
        pass

    def evaluate_adaptability(self):
        pass
    
    def do_evaluations(self, todo_dict=None):
        """
        this function will call all the evaluation functions 
        """

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
