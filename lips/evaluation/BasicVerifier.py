# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import numpy as np
from collections import Counter
from sklearn.metrics import mean_absolute_error
import sys
import logging

# TODO make a proper class there and that inside ! It's super ugly not to be able to customize this
logging.basicConfig(filename="logs.log",
                    level=logging.INFO,
                    format="%(levelname)s:%(message)s")


def BasicVerifier(active_dict: dict,
                  predictions=None,
                  line_status=None 
                  ):
    """
    verify the following elementary physic compliances

    - verify that the voltages are greater than zero in both extremity of power lines v_or >= 0 , v_ex >= 0
    - verify that the currents are greater than zero in both extremity of power lines a_or >= 0 et a_ex >= 0
    - verify that the electrical losses are greater than zero at each power line p_or + p_ex >= 0
    - verify if a line is disconnected the corresponding flux (p, q and a) are zero
    - verify the following relation between p, q and v : 
        * a_or = sqrt(p_or**2 + q_or**2) / (sqrt(3).v_or) 
        * a_ex = sqrt(p_ex**2 + q_ex**2) / (sqrt(3).v_ex)

    params
    ------
        p_or: `array`
            the active power at the origin side of power lines

        p_ex: `array`
            the active power at the extremity side of power lines

        q_or: `array`
            the reactive power at the origin side of power lines

        p_ex: `array`
            the reactive power at the extremity side of power lines

        a_or: `array`
            the current at the origin side of power lines

        a_ex: `array`
            the current at the extremity side of power lines

        v_or: `array`
            the voltage at the origin side of power lines

        v_ex: `array`
            the voltage at the extremity side of power lines

        line_status: `array`
            the line_status vector at each iteration presenting the connectivity status of power lines


    Returns
    -------
    a dict with a check list of verified points for each observation

    """
    #print("************* Basic verifier *************")
    verifications = dict()

    # VERIFICATION 1
    # verification of currents
    if active_dict["verify_current_pos"]:
        a_or = predictions["a_or"]
        a_ex = predictions["a_ex"]
        verifications["currents"] = {}
        if np.any(a_or < 0):
            verifications["currents"]["a_or"] = {}
            a_or_errors = np.array(np.where(a_or < 0)).T
            a_or_violation_proportion = len(a_or_errors) / a_or.size
            Error_a_or = -np.sum(np.minimum(a_or.flatten(), 0.))
            #print("{:.3f}% of lines does not respect the positivity of currents (Amp) at origin".format(
            #    a_or_violation_proportion*100))
            logging.info("the sum of negative current values (A) at origin: {:.3f}".format(Error_a_or))
            # print the concerned lines with the counting their respectives anomalies
            counts = Counter(a_or_errors[:, 1])
            #print("Concerned lines with corresponding number of negative current values at their origin:\n",
            #    dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
            verifications["currents"]["a_or"]["indices"] = a_or_errors
            verifications["currents"]["a_or"]["Error"] = Error_a_or
            verifications["currents"]["a_or"]["Violation_proportion"] = a_or_violation_proportion
            #print("----------------------------------------------")
        else:
            logging.info("Current positivity check passed for origin side !")
            #print("Current positivity check passed for origin side !")
            #print("----------------------------------------------")

        if np.any(a_ex < 0):
            verifications["currents"]["a_ex"] = {}
            a_ex_errors = np.array(np.where(a_ex < 0)).T
            a_ex_violation_proportion = len(a_ex_errors) / a_ex.size
            Error_a_ex = -np.sum(np.minimum(a_ex.flatten(), 0.))
            #print("{:.3f}% of lines does not respect the positivity of currents (Amp) at extremity".format(
            #    a_ex_violation_proportion*100))
            logging.info("the sum of negative current values (A) at extremity: {:.3f}".format(Error_a_ex))
            # print the concerned lines with the counting their respectives anomalies
            counts = Counter(a_ex_errors[:, 1]) 
            #print("Concerned lines with corresponding number of negative current values at their extremity:\n",
            #    dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
            verifications["currents"]["a_ex"]["indices"] = a_ex_errors
            verifications["currents"]["a_ex"]["Error"] = Error_a_ex
            verifications["currents"]["a_ex"]["Violation_proportion"] = a_ex_violation_proportion
            #print("----------------------------------------------")
        else:
            logging.info("Current positivity check passed for extremity side !")
            #print("Current positivity check passed for extremity side !")
            #print("----------------------------------------------")

    # VERIFICATION 2
    # verification of voltages
    if active_dict["verify_voltage_pos"]:
        v_or = predictions["v_or"]
        v_ex = predictions["v_ex"]
        verifications["voltages"] = {}
        if np.any(v_or < 0):
            verifications["voltages"]["v_or"] = {}
            v_or_errors = np.array(np.where(v_or < 0)).T
            v_or_violation_proportion = len(v_or_errors) / v_or.size
            Error_v_or = -np.sum(np.minimum(v_or.flatten(), 0.))
            logging.info("the sum of negative voltage values (kV) at origin: {:.3f}".format(Error_v_or))
            #print("{:.3f}% of lines does not respect the positivity of voltages (Kv) at origin".format(
            #    v_or_violation_proportion*100))
            # print the concerned lines with the counting their respectives anomalies
            counts = Counter(v_or_errors[:, 1])
            #print("Concerned lines with corresponding number of negative voltage values at their origin:\n",
            #    dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
            verifications["voltages"]["v_or"]["indices"] = v_or_errors
            verifications["voltages"]["v_or"]["Error"] = Error_v_or
            verifications["voltages"]["v_or"]["Violation_proportion"] = v_or_violation_proportion
            #print("----------------------------------------------")
        else:
            logging.info("Voltage positivity check passed for origin side !")
            #print("Voltage positivity check passed for origin side !")
            #print("----------------------------------------------")

        if np.any(v_ex < 0):
            verifications["voltages"]["v_ex"] = {}
            v_ex_errors = np.array(np.where(v_ex < 0)).T
            v_ex_violation_proportion = len(v_ex_errors) / v_ex.size
            Error_v_ex = -np.sum(np.minimum(v_ex.flatten(), 0.))
            logging.info("the sum of negative voltage values (kV) at extremity: {:.3f}".format(Error_v_ex))
            #print("{:.3f}% of lines does not respect the positivity of voltages (Kv) at extremity".format(
            #    v_ex_violation_proportion*100))
            # print the concerned lines with the counting their respectives anomalies
            counts = Counter(v_ex_errors[:, 1])
            #print("Concerned lines with corresponding number of negative voltage values at their extremity:\n",
            #    dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
            verifications["voltages"]["v_ex"]["indices"] = v_ex_errors
            verifications["voltages"]["v_ex"]["Error"] = Error_v_ex
            verifications["voltages"]["v_ex"]["Violation_proportion"] = v_ex_violation_proportion
            #print("----------------------------------------------")
        else:
            logging.info("Voltage positivity check passed for extremity side !")
            #print("Voltage positivity check passed for extremity side !")
            #print("----------------------------------------------")

    # VERIFICATION 3
    # Positivity of losses
    if active_dict["verify_loss_pos"]:
        p_or = predictions["p_or"]
        p_ex = predictions["p_ex"]

        verifications["loss"] = {}
        loss = p_or + p_ex

        if np.any(loss):
            loss_error = -np.sum(np.minimum(loss, 0.))
            loss_errors = np.array(np.where(loss < 0)).T
            loss_violation_proportion = len(loss_errors) / p_or.size
            logging.info("the sum of negative losses : {:.3f}".format(loss_error))
            #print("{:.3f}% of lines does not respect the positivity of loss (Mw)".format(
            #    loss_violation_proportion*100))
            # print the concerned lines with the counting their respectives anomalies
            counts = Counter(loss_errors[:, 1])
            #print("Concerned lines with corresponding number of negative loss values:\n",
            #    dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
            verifications["loss"]["loss_criterion"] = loss_error
            verifications["loss"]["loss_errors"] = loss_errors
            verifications["loss"]["violation_proportion"] = loss_violation_proportion
            #print("----------------------------------------------")
        else:
            logging.info("Loss positivity check passed !")
            #print("Loss positivity check passed !")
            #print("----------------------------------------------")

    # VERIFICATION 4
    # verifying null values for line disconnections
    if active_dict["verify_predict_disc"]:

        verifications["line_status"] = {}
        sum_disconnected_values = 0

        ind_ = line_status != 1
        len_disc = np.sum(ind_)
        if np.any(ind_):
            if "p_or" in predictions.keys():
                p_or = predictions["p_or"]
                p_ex = predictions["p_ex"]
                p_or_violations = np.sum(np.abs(p_or[ind_]) > 0)
                verifications["line_status"]["p_or_not_null"] = p_or_violations
                sum_disconnected_values += p_or_violations
                p_ex_violations = np.sum(np.abs(p_ex[ind_]) > 0)
                verifications["line_status"]["p_ex_not_null"] = p_ex_violations
                sum_disconnected_values += p_ex_violations
                verifications["line_status"]["p_violations"] = np.sum((np.abs(p_or[ind_]) + np.abs(p_ex[ind_]))>0) / len_disc
            if "q_or" in predictions.keys():
                q_or = predictions["q_or"]
                q_ex = predictions["q_ex"]
                q_or_violations = np.sum(np.abs(q_or[ind_]) > 0)
                verifications["line_status"]["q_or_not_null"] = q_or_violations
                sum_disconnected_values += q_or_violations
                q_ex_violations = np.sum(np.abs(q_ex[ind_]) > 0)
                verifications["line_status"]["q_ex_not_null"] = q_ex_violations
                sum_disconnected_values += q_ex_violations
                verifications["line_status"]["q_violations"] = np.sum((np.abs(q_or[ind_]) + np.abs(q_ex[ind_]))>0) / len_disc
            if "a_or" in predictions.keys():
                a_or = predictions["a_or"]
                a_ex = predictions["a_ex"]
                a_or_violations = np.sum(np.abs(a_or[ind_]) > 0)
                verifications["line_status"]["a_or_not_null"] = a_or_violations
                sum_disconnected_values += a_or_violations
                a_ex_violations = np.sum(np.abs(a_ex[ind_]) > 0)
                verifications["line_status"]["a_ex_not_null"] = a_ex_violations
                sum_disconnected_values += a_ex_violations
                verifications["line_status"]["a_violations"] = np.sum((np.abs(a_or[ind_]) + np.abs(a_ex[ind_]))>0) / len_disc
        if sum_disconnected_values > 0:
            logging.info("Prediction in presence of line disconnection. Problem encountered !")
            #print("Prediction in presence of line disconnection. Problem encountered !")
        else:
            #print("Prediction in presence of line disconnection. Check passed !")
            logging.info("Prediction in presence of line disconnection. Check passed !")    
        #print("----------------------------------------------")

    # VERIFICATION 5 and 6
    # Verify current equations for real and predicted observations
    # TODO : update the equations by considering only voltage > 0 cases, hence it does not need eps
    if active_dict["verify_current_eq"]:
        a_or = predictions["a_or"]
        a_ex = predictions["a_ex"]
        p_or = predictions["p_or"]
        p_ex = predictions["p_ex"]
        q_or = predictions["q_or"]
        q_ex = predictions["q_ex"]
        v_or = predictions["v_or"]
        v_ex = predictions["v_ex"]

        verifications["current_equations"] = {}
        # consider an epsilon value to avoid division by zero
        eps = sys.float_info.epsilon
        #a_or = sqrt(p_or**2 + q_or**2) / (sqrt(3).v_or)
        a_or_comp = (np.sqrt(p_or**2 + q_or**2) / ((np.sqrt(3) * v_or)+eps)) * 1000
        verifications["current_equations"]["a_or_deviation"] = mean_absolute_error(
            a_or, a_or_comp, multioutput='raw_values')

        #a_ex = sqrt(p_ex**2 + q_ex**2) / (sqrt(3).v_ex)
        a_ex_comp = (np.sqrt(p_ex**2 + q_ex**2) / ((np.sqrt(3) * v_ex)+eps)) * 1000
        verifications["current_equations"]["a_ex_deviation"] = mean_absolute_error(
            a_ex, a_ex_comp, multioutput='raw_values')

    return verifications
