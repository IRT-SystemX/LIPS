"""
Function to compute the Joule law for power grid use case
"""
from typing import Union
import numpy as np

from lightsim2grid import LightSimBackend
from grid2op.Backend import PandaPowerBackend

from ...logger import CustomLogger

def verify_joule_law(predictions: dict,
                     log_path: Union[str, None]=None,
                     result_level: int=0,
                     **kwargs):
    """
    This function compute Joule's law which is ``$PL = R \times I^2$``

    To compute the PowerLoss(PL), we use : ``$p_ex + p_or$``
    The resistance values are deduced from backends.
    The currents are computed as follows : ``$\frac{a_or + a_ex}{2}$``

    TODO : consider the line disconnections

     Parameters
    ----------
    predictions: ``dict``
        Predictions made by an augmented simulator
    log_path: ``str``, optional
        the path where the logs should be saved, by default None
    **kwargs: ``dict``
        It should contain `observations` and `config` to load the tolerance

    Returns
    -------
    ``dict``
        a dictionary with useful information about the verification

    These informations are:
    - mae_per_obs: `array`
        The Absolute error (difference) between two sides of Joule equation per observation

    - mae_per_obj: `array`
        The Absolute Error (difference) between two sides of Joule equation per line for each observation

    - mae: `scalar`
        the Mean Absolute Error (difference) between two sides of the Joule equation

    - wmape: `scalar`
        weighted mean absolute percentage error between two sides of the Joule equation

    - violation_proportion: `scalar`
        The percentage of violation of Joule law over all the observations
    """
     # logger
    logger = CustomLogger("PhysicsCompliances(Joule_law)", log_path).logger

    try:
        env = kwargs["env"]
    except KeyError:
        logger.error("The requirements were not satisiftied to verify_joule_law function")
        raise

    try:
        config = kwargs["config"]
    except KeyError:
        try:
            tolerance = kwargs["tolerance"]
        except KeyError:
            logger.error("The tolerance could not be found for verify_joule_law function")
            raise
        else:
            tolerance = float(tolerance)
    else:
        tolerance = float(config.get_option("eval_params")["JOULE_tolerance"])

    if isinstance(env.backend, LightSimBackend):
        # extract the impedance values which are in p.u
        z_pu = [el.r_pu for el in env.backend._grid.get_lines()]
        z_pu_trafo = [el.r_pu for el in env.backend._grid.get_trafos()]
        # enumerate lines and trafos
        n_lines = len(z_pu)
        n_trafo = len(z_pu_trafo)
        # transform pu to Ohm and keep only lines
        z_base = np.power(env.backend.lines_or_pu_to_kv, 2) / env.backend._grid.get_sn_mva()
        r_ohm = z_pu * z_base[:n_lines]

    elif isinstance(env.backend, PandaPowerBackend):
        # n_lines
        n_lines = len(env.backend._grid.line)
        # n_transformers
        n_transformers = len(env.backend._grid.trafo)
        # get the resistance
        r_ohm = np.array(env.backend._grid.line["r_ohm_per_km"]).reshape(1,-1)

    verifications = dict()
    mean_current = _get_average_currents(predictions)
    mean_current_squared = np.power(mean_current, 2)
    pl_mw = _get_power_loss(predictions)
    # MAE between PL and R.I^2
    left_array = pl_mw[:, :n_lines]/3
    right_array = mean_current_squared[:, :n_lines] * r_ohm
    mae_per_obj = np.abs(left_array - right_array)
    mae_per_obs = mae_per_obj.mean(axis=1)
    mae = np.mean(mae_per_obs)
    wmape = np.mean(np.abs(left_array - right_array)) / np.mean(np.abs(left_array))
    violation_prop = np.sum(mae_per_obj > tolerance) / mae_per_obj.size

    logger.info("Joule law violation proportion: %.3f", violation_prop)
    logger.info("MAE for Joule law: %.3f", mae)
    logger.info("WMAPE for Joule law: %.3f", wmape)

    if result_level > 0:
        verifications["mae_per_obs"] = mae_per_obs
        verifications["mae_per_obj"] = mae_per_obj
    verifications["violation_proportion"] = violation_prop
    verifications["mae"] = mae
    verifications["wmape"] = wmape

    return verifications

def _get_average_currents(data: dict) -> np.ndarray:
    """Compute the average current using current from both extremities of a power line

    Parameters
    ----------
    data : ``dict``
        A python dictionary including currents as keys

    Returns
    -------
    np.array
        average matrix of currents
    """
    current_avg = np.abs(data.get("a_or") + data.get("a_ex"))/2
    current_avg_kA = current_avg / 1000
    return current_avg_kA

def _get_power_loss(data: dict) -> np.ndarray:
    """Compute the power loss from powers at two extremities of power lines

    Parameters
    ----------
    data : ``dict``
        A python dictionary including powers as keys

    Returns
    -------
    ``np.ndarray``
        Power losses array at the line level for all the observations
    """
    pl_mw = np.abs(data.get("p_or") + data.get("p_ex"))
    return pl_mw