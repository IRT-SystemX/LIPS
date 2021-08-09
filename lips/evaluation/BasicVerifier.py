# Pour préciser ce qu'entendais Antoine par les derniers points (si on demande aux gens de prédire v_or, v_ex, a_or, a_ex, p_or, p_ex et q_or, q_ex):
# - sur chaque ligne, v_or >= 0 , v_ex >= 0
# - sur chaque ligne a_or >= 0 et a_ex >= 0
# - sur chaque ligne, les pertes doivent être positives: p_or + p_ex >= 0
# - on a une relation entre les p,q et v à chaque extrémité de ligne connectée : a_or = sqrt(p_or**2 + q_or**2) / (sqrt(3).v_or) et a_ex = sqrt(p_ex**2 + q_ex**2) / (sqrt(3).v_ex)
# - si une ligne est déconnecté, alors son flux (en p, en q et en a) à chaque bout (donc pour _or et _ex) est nul (6 choses à vérifier)

import numpy as np
from collections import Counter
from sklearn.metrics import mean_absolute_error
import sys


def BasicVerifier(a_or, a_ex, v_or, v_ex, p_or, p_ex, q_or, q_ex, line_status):
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
    print("************* Basic verifier *************")
    verifications = dict()

    # VERIFICATION 1
    # verification of currents
    verifications["currents"] = {}
    if np.any(a_or < 0):
        a_or_errors = np.array(np.where(a_or < 0)).T
        print("{:.3f}% of lines does not respect the positivity of currents (Amp) at origin".format(
            (len(a_or_errors) / a_or.size)*100))
        # print the concerned lines with the counting their respectives anomalies
        counts = Counter(a_or_errors[:, 1])
        print("Concerned lines with corresponding number of negative current values at their origin:\n",
              dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
        verifications["currents"]["a_or_errors"] = a_or_errors
        print("----------------------------------------------")
    else:
        print("Current positivity check passed for origin side !")
        print("----------------------------------------------")

    if np.any(a_ex < 0):
        a_ex_errors = np.array(np.where(a_ex < 0)).T
        print("{:.3f}% of lines does not respect the positivity of currents (Amp) at extremity".format(
            (len(a_ex_errors) / a_ex.size)*100))
        # print the concerned lines with the counting their respectives anomalies
        counts = Counter(a_ex_errors[:, 1])
        print("Concerned lines with corresponding number of negative current values at their extremity:\n",
              dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
        verifications["currents"]["a_ex_errors"] = a_ex_errors
        print("----------------------------------------------")
    else:
        print("Current positivity check passed for extremity side !")
        print("----------------------------------------------")

    # VERIFICATION 2
    # verification of voltages
    verifications["voltages"] = {}
    if np.any(v_or < 0):
        v_or_errors = np.array(np.where(v_or < 0)).T
        print("{:.3f}% of lines does not respect the positivity of voltages (Kv) at origin".format(
            (len(v_or_errors) / v_or.size)*100))
        # print the concerned lines with the counting their respectives anomalies
        counts = Counter(v_or_errors[:, 1])
        print("Concerned lines with corresponding number of negative voltage values at their origin:\n",
              dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
        verifications["voltages"]["v_or_errors"] = v_or_errors
        print("----------------------------------------------")
    else:
        print("Voltage positivity check passed for origin side !")
        print("----------------------------------------------")

    if np.any(v_ex < 0):
        v_ex_errors = np.array(np.where(v_ex < 0)).T
        print("{:.3f}% of lines does not respect the positivity of voltages (Kv) at extremity".format(
            (len(v_ex_errors) / v_ex.size)*100))
        # print the concerned lines with the counting their respectives anomalies
        counts = Counter(v_ex_errors[:, 1])
        print("Concerned lines with corresponding number of negative voltage values at their extremity:\n",
              dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
        verifications["voltages"]["v_ex_errors"] = v_ex_errors
        print("----------------------------------------------")
    else:
        print("Voltage positivity check passed for extremity side !")
        print("----------------------------------------------")

    # VERIFICATION 3
    # Positivity of losses
    verifications["loss"] = {}
    loss = p_or + p_ex

    if np.any(loss):
        verifications["loss"]["loss_criterion"] = -np.sum(np.minimum(loss, 0.))
        loss_errors = np.array(np.where(loss < 0)).T
        print("{:.3f}% of lines does not respect the positivity of loss (Mw)".format(
            (len(loss_errors) / p_or.size)*100))
        # print the concerned lines with the counting their respectives anomalies
        counts = Counter(loss_errors[:, 1])
        print("Concerned lines with corresponding number of negative loss values:\n",
              dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))
        verifications["loss"]["loss_errors"] = loss_errors
        print("----------------------------------------------")
    else:
        print("Loss positivity check passed !")
        print("----------------------------------------------")

    # VERIFICATION 4
    # verifying null values for line disconnections
    verifications["line_status"] = {}
    sum_disconnected_values = 0

    ind_ = line_status != 1
    if np.any(ind_):
        verifications["line_status"]["p_or_not_null"] = np.sum(p_or[ind_] > 0)
        sum_disconnected_values += np.sum(p_or[ind_] > 0)
        verifications["line_status"]["p_ex_not_null"] = np.sum(p_ex[ind_] > 0)
        sum_disconnected_values += np.sum(p_ex[ind_] > 0)
        verifications["line_status"]["q_or_not_null"] = np.sum(q_or[ind_] > 0)
        sum_disconnected_values += np.sum(q_or[ind_] > 0)
        verifications["line_status"]["q_ex_not_null"] = np.sum(q_ex[ind_] > 0)
        sum_disconnected_values += np.sum(q_ex[ind_] > 0)
        verifications["line_status"]["a_or_not_null"] = np.sum(a_or[ind_] > 0)
        sum_disconnected_values += np.sum(a_or[ind_] > 0)
        verifications["line_status"]["a_ex_not_null"] = np.sum(a_ex[ind_] > 0)
        sum_disconnected_values += np.sum(a_ex[ind_] > 0)
    if sum_disconnected_values > 0:
        print("Prediction in presence of line disconnection. Problem encountered !")
    else:
        print("Prediction in presence of line disconnection. Check passed !")
    print("----------------------------------------------")

    # VERIFICATION 5 and 6
    # Verify current equations for real and predicted observations
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
