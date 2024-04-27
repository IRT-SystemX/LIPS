"""
Functions to compute the equality of voltages at each bus of the powergrid
"""
from typing import Union
from tqdm import tqdm
import numpy as np

import grid2op
from grid2op import Observation
from ...dataset.utils.powergrid_utils import get_kwargs_simulator_scenario
from ...logger import CustomLogger


# helper functions
def _get_fake_obs(config) -> Observation:
    env = grid2op.make(**get_kwargs_simulator_scenario(config))
    env.deactivate_forecast()
    obs, *_ = env.step(env.action_space({}))
    return obs

def _aux_get_buses(obs):
    """
    the part extracted from flow_bus_matrix of observation

    params
    ------
        obs : `observation`
            a fake observations updated using topology configuration at each step

    Return
    ------
        load_bus: `list`
            the indices of buses that loads are connected to

        prod_bus: `list`
            the indices of buses that generators are connected to

        stor_bus: `list`
            the indices of buses that storages are connected to

        lor_bus: `list`
            the indices of buses that origin side of lines are connected to

        lex_bus: `list`
            the indices of buses that extremity side of lines are connected to
    """
    nb_bus, unique_bus, bus_or, bus_ex = obs._aux_fun_get_bus()
    prod_bus, prod_conn = obs._get_bus_id(obs.gen_pos_topo_vect, obs.gen_to_subid)
    load_bus, load_conn = obs._get_bus_id(obs.load_pos_topo_vect, obs.load_to_subid)
    stor_bus, stor_conn = obs._get_bus_id(obs.storage_pos_topo_vect, obs.storage_to_subid)
    lor_bus, lor_conn = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
    lex_bus, lex_conn = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)

    if obs.shunts_data_available:
        sh_bus = 1 * obs._shunt_bus
        sh_bus[sh_bus > 0] = obs.shunt_to_subid[sh_bus > 0]*(sh_bus[sh_bus > 0] - 1) + \
                             obs.shunt_to_subid[sh_bus > 0]
        sh_conn = obs._shunt_bus != -1

    # convert the bus to be "id of row or column in the matrix" instead of the bus id with
    # the "grid2op convention"
    all_indx = np.arange(nb_bus)
    tmplate = np.arange(np.max(unique_bus)+1)
    tmplate[unique_bus] = all_indx
    prod_bus = tmplate[prod_bus]
    load_bus = tmplate[load_bus]
    lor_bus = tmplate[lor_bus]
    lex_bus = tmplate[lex_bus]
    stor_bus = tmplate[stor_bus]

    # added to avoid considering disconnected lines
    line_disc = np.where(obs.line_status==False)[0]
    lor_bus[line_disc] = -1
    lex_bus[line_disc] = -1

    return load_bus, prod_bus, stor_bus, lor_bus, lex_bus

def _aux_retrieve_elements_at_bus(obs, bus_id=0):
    """
    Returns the relative indices of buses that each element is connected to
    """
    load_bus, gen_bus, stor_bus, lor_bus, lex_bus = _aux_get_buses(obs)
    #load_bus_aux = obs.load_to_subid
    #gen_bus_aux = obs.gen_to_subid

    lor_bus_idx = np.where(lor_bus==bus_id)[0]
    lex_bus_idx = np.where(lex_bus==bus_id)[0]
    gen_bus_idx = np.where(gen_bus==bus_id)[0]
    load_bus_idx = np.where(load_bus==bus_id)[0]
    stor_bus_idx = np.where(stor_bus==bus_id)[0]

    return (lor_bus_idx, lex_bus_idx), gen_bus_idx, load_bus_idx, stor_bus_idx

def _aux_get_voltages_at_bus(obs, bus_id, return_theta=True):
    """
    this function gets the voltages and voltage angles for an indicated bus from data

    params
    ------
        obs: ``object`` of grid2op ``observation``

        data: ``dict``
            a dictionary including the variables for all the observations

        bus_id: ``int``
            the bus_id for which we would like to retrieve the voltages

        return_theta: ``bool``
            whether to retrieve the voltage angle (theta) values
    """
    (lor_bus_idx, lex_bus_idx), gen_bus_idx, load_bus_idx, stor_bus_idx = _aux_retrieve_elements_at_bus(obs, bus_id=bus_id)
    v_or_sub = obs.v_or[lor_bus_idx]#data["v_or"][:, lines_or_bus]
    v_ex_sub = obs.v_ex[lex_bus_idx]#data["v_ex"][:, lines_ex_bus]
    load_v_sub = obs.load_v[load_bus_idx]#data["load_v"][:, load_bus]
    gen_v_sub = obs.gen_v[gen_bus_idx]#data["prod_v"][:, gen_bus]

    voltages = (v_or_sub, v_ex_sub, gen_v_sub, load_v_sub)

    thetas = None
    if return_theta:
        theta_or_sub = obs.theta_or[lor_bus_idx] #data["theta_or"][:, lines_or_bus]
        theta_ex_sub = obs.theta_ex[lex_bus_idx] #data["theta_ex"][:, lines_ex_bus]
        load_theta_sub = obs.load_theta[load_bus_idx] #data["load_theta"][:, load_bus]
        gen_theta_sub = obs.gen_theta[gen_bus_idx] #data["gen_theta"][:, gen_bus]
        thetas = (theta_or_sub, theta_ex_sub, gen_theta_sub, load_theta_sub)

    return voltages, thetas

def _aux_update_observation(obs, real_data, predictions, idx):
    """
    this function replaces all the variables of observation with those included in data
    """
    # a set of attributes available in ref_obs but not in predictions (injections + topology)
    # it concerns the attributs which are used as predictors
    real_data_keys = real_data.keys() - predictions.keys()
    # replacing the `obs` attributes using the injections + topology (real data)
    for attr_nm in real_data_keys:
        if attr_nm == "prod_v":
            attr_nm_obs = "gen_v"
        elif attr_nm == "prod_q":
            attr_nm_obs = "gen_q"
        elif attr_nm == "prod_p":
            attr_nm_obs = "gen_p"
        elif attr_nm == "storage_theta":
            attr_nm_obs = ""
        else:
            attr_nm_obs = attr_nm
        setattr(obs, attr_nm_obs, real_data[attr_nm][idx])

    # replacing the remaining attributes using the predictions
    for attr_nm, val_ in predictions.items():
        if attr_nm == "prod_v":
            attr_nm = "gen_v"
        elif attr_nm == "prod_q":
            attr_nm = "gen_q"
        elif attr_nm == "prod_p":
            attr_nm = "gen_p"
        elif attr_nm == "storage_theta":
            attr_nm = ""
        setattr(obs, attr_nm, val_[idx])

    return obs

#def verify_voltage_at_bus(obs, real_data, predictions, tol=1e-4, verify_theta=True):
def verify_voltage_at_bus(predictions: dict,
                          log_path: Union[str, None]=None,
                          result_level: int=0,
                          **kwargs):
    """
    This functions checks if the elements connected to a same substations present the same voltages and angles

    return
    ------
    The result of this verification are presented in two fashions:
    - the rate of not respected voltage values (in terms of proportion of substations)
    - the mean and standard deviation of voltage values at each node
    - the MAE between voltage values (it approaches the MAE computed over all the observations)
    """
    # logger
    logger = CustomLogger("PhysicsCompliances(voltage_eq)", log_path).logger
    try:
        observations = kwargs["observations"]
    except KeyError:
        logger.error("The requirements were not satisiftied to call verify_voltage_at_bus function")
        raise
    try:
        config = kwargs["config"]
    except KeyError:
        try:
            tol = kwargs["tolerance"]
        except KeyError:
            logger.error("The tolerance could not be found for verify_voltage_at_bus function")
            raise
        else:
            tol = float(tol)
    else:
        tol = float(config.get_option("eval_params")["VOLTAGE_EQ"]["tolerance"])
        verify_theta = bool(config.get_option("eval_params")["VOLTAGE_EQ"]["verify_theta"])

    verifications = dict()
    n_obs = len(observations["a_or"])
    obs = _get_fake_obs(config)
    n_buses = 2 * obs.n_sub

    mean_matrix_voltage = np.zeros((n_obs, n_buses), dtype=float)
    mean_matrix_voltage[:] = np.NaN
    std_matrix_voltage = np.zeros((n_obs, n_buses), dtype=float)
    std_matrix_voltage[:] = np.NaN
    mean_matrix_theta = np.zeros((n_obs, n_buses), dtype=float)
    mean_matrix_theta[:] = np.NaN
    std_matrix_theta = np.zeros((n_obs, n_buses), dtype=float)
    std_matrix_theta[:] = np.NaN


    for i in tqdm(range(n_obs)):
        # update the observation with respect to data topology
        # this step is required, because the connectivity matrices and vectors evolve wrt. topology changes
        obs_updated = _aux_update_observation(obs, observations, predictions, i)


        for bus_ in range(n_buses):
            # extract the voltages and thetas connected to same ``bus_id`` busbar of the network
            voltages, thetas = _aux_get_voltages_at_bus(obs_updated, bus_id=bus_, return_theta=verify_theta)
            voltages_at_sub = np.hstack(voltages)

            if len(voltages_at_sub) > 0:
                mean_matrix_voltage[i, bus_] = voltages_at_sub.mean()
                std_matrix_voltage[i, bus_] = voltages_at_sub.std()

            if verify_theta:
                thetas_at_sub = np.hstack(thetas)
                if len(thetas_at_sub) > 0:
                    mean_matrix_theta[i, bus_] = thetas_at_sub.mean()
                    std_matrix_theta[i, bus_] = thetas_at_sub.std()

    #prop_voltages_violation = np.sum(np.nansum(std_matrix_voltage, axis=1) > tol) / len(std_matrix_voltage)
    #prop_theta_violation = np.sum(np.nansum(std_matrix_theta, axis=1) > tol) / len(std_matrix_theta)
    prop_voltages_violation = np.sum(std_matrix_voltage[~np.isnan(std_matrix_voltage)] > tol) / np.sum(~np.isnan(std_matrix_voltage))
    if verify_theta:
        prop_theta_violation = np.sum(std_matrix_theta[~np.isnan(std_matrix_theta)] > tol) / np.sum(~np.isnan(std_matrix_theta))
    else:
        prop_theta_violation = None

    verifications["prop_voltages_violation"] = prop_voltages_violation
    if prop_theta_violation is not None:
        verifications["prop_theta_violation"] = prop_theta_violation

    if result_level > 0:
        verifications["mean_matrix_voltage"] = mean_matrix_voltage
        verifications["std_matrix_voltage"] = std_matrix_voltage
        if prop_theta_violation is not None:
            verifications["mean_matrix_theta"] = mean_matrix_theta
            verifications["std_matrix_theta"] = std_matrix_theta


    return verifications
    #return (mean_matrix_voltage, std_matrix_voltage, prop_voltages_violation), (mean_matrix_theta, std_matrix_theta, prop_theta_violation)
