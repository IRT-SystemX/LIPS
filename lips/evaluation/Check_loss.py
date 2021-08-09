import numpy as np
import copy
import os


def Check_loss(p_or, p_ex, prod_p, tolerance=0.04):
    """
    Verifies the energy loss. The loss should be between 1 and 4 % of production at each step.

    2 possible way to call the function with two set of informations : 
        1) indicating only the path to the stored arrays by using the `path` parameter
        2) indicating explicitly the required variables for computing the law which are (prod_p, p_or and p_ex)

    params
    ------
        prod_p: `array`
            the active power of production points in power network

        p_or: `array`
            the active power of the origin side of power lines

        p_ex: `array`
            the active power of the extremity side of power lines

        tolerance: `float`
            a threshold value used to compare the value of the law, and below which the verification is failed

    Returns
    -------
        EL: `array`
            array of energy losses for each iteration

        failed_indices: `list`
            The indices of failed cases
    """
    print("************* Check loss *************")
    failed_indices = None
    violation_percentage = None

    EL = np.abs(np.sum(p_or + p_ex, axis=1))
    ratio = EL / np.sum(prod_p, axis=1)

    condition = ratio > tolerance
    if np.any(condition):
        failed_indices = np.array(np.where(condition)).reshape(-1, 1)
        violation_percentage = (len(failed_indices) / len(EL))*100
        print("Number of failed cases is {} and the proportion is {:.3f}% : ".format(len(
            failed_indices), violation_percentage))
    else:
        print("Verification is done without any violation !")
        violation_percentage = 0.

    return EL, violation_percentage, failed_indices
