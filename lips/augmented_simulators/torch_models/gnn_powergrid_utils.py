from typing import Union
from cmath import pi
import numpy as np
import numpy.typing as npt

import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader

from grid2op import Observation

def prepare_dataset(obs, dataset, return_edge_weights=False):
    features = get_all_features_per_sub_v2(obs, dataset)
    targets = get_target_variables_per_sub(obs, dataset)
    edge_indices = get_edge_index_from_ybus(dataset["YBus"])
    if return_edge_weights:
        edge_weights = get_edge_weights_from_ybus(dataset["YBus"], edge_indices)
    else:
        edge_weights = None
    
    return features, targets, edge_indices, edge_weights

def get_features_per_sub_v2(obs: Observation, dataset: dict, index: int=0) -> npt.NDArray[np.float32]:
    """It returns the features at substation level without aggregation

    Parameters
    ----------
    obs : ``Observation``
        Grid2op observation
    dataset : ``DataSet``
        _description_
    index : ``int``, optional
        A specific index of dataset for which the features should be extracted, by default 0

    Returns
    -------
    npt.NDArray[np.float32]
        the feature array for one observation of dataset
    """
    feature_matrix = np.zeros((obs.n_sub, 2), dtype=np.float32)
    for sub_ in range(obs.n_sub):
        objects = obs.get_obj_connect_to(substation_id=sub_)

        if len(objects["generators_id"]) > 0:
            feature_matrix[sub_, 0] += np.sum(dataset.get("prod_p")[index, objects["generators_id"]])
        if len(objects["loads_id"]) > 0:
            feature_matrix[sub_,1] += np.sum(dataset.get("load_p")[index, objects["loads_id"]])
        
    return feature_matrix  

def get_all_features_per_sub_v2(obs: Observation, dataset: dict) -> torch.Tensor:
    """Get all the features from dataset without their aggregation

    Parameters
    ----------
    obs : Observation
        Grid2op observation used for some functionalities
    dataset : DataSet
        LIPS dataset from which the features should be extracted

    Returns
    -------
    torch.Tensor
        Torch tensor including the features that should be used as the inputs for a model
    """
    features = torch.zeros((len(dataset["prod_p"]), obs.n_sub, 2))
    for i in range(len(features)):
        features[i, :, :] = torch.tensor(get_features_per_sub_v2(obs, dataset, index=i))
    return features.float()

def create_fake_obs(obs, data, idx = 0):
    obs.line_status = data["line_status"][idx]
    obs.topo_vect = data["topo_vect"][idx]
    return obs

def create_fake_obs_custom(obs, line_status, topo_vect, idx = 0):
    obs.line_status = line_status[idx]
    obs.topo_vect = topo_vect[idx]
    return obs

def get_theta_node(obs, sub_id, bus):
    obj_to_sub = obs.get_obj_connect_to(substation_id=sub_id)

    lines_or_to_sub_bus = [i for i in obj_to_sub['lines_or_id'] if obs.line_or_bus[i] == bus]
    lines_ex_to_sub_bus = [i for i in obj_to_sub['lines_ex_id'] if obs.line_ex_bus[i] == bus]

    thetas_node = np.append(obs.theta_or[lines_or_to_sub_bus], obs.theta_ex[lines_ex_to_sub_bus])
    thetas_node = thetas_node[thetas_node != 0]

    theta_node = 0.
    if len(thetas_node) != 0:
        #theta_node = np.median(thetas_node)
        theta_node = np.max(thetas_node)

    return theta_node

def get_theta_bus(dataset, obs):
    """
    Function to compute complex voltages at bus
    """
    Ybus = dataset["YBus"]
    # these two lines could be uncommented
    #bus_theta = np.zeros((Ybus.shape[0], Ybus.shape[1]), dtype=complex)
    #for idx in range(Ybus.shape[0]):
    bus_theta = np.zeros((Ybus.shape[0], obs.n_sub), dtype=complex)
    for idx in range(Ybus.shape[0]):
        #obs.topo_vect = dataset["topo_vect"][idx]
        obs = create_fake_obs(obs, dataset, idx)
        obs.theta_or = dataset["theta_or"][idx]
        obs.theta_ex = dataset["theta_ex"][idx]
        for sub_ in range(obs.n_sub):
            bus_theta[idx, sub_] = get_theta_node(obs, sub_id=sub_, bus=1)
        
    return bus_theta

def get_target_variables_per_sub(obs: Observation, dataset: dict) -> torch.Tensor:
    """Gets the target variables which should be predicted by a model

    Parameters
    ----------
    obs : Observation
        Grid2op observation
    dataset : DataSet
        LIPS dataset from which the target variable(s) could be extracted

    Returns
    -------
    torch.Tensor
        Tensor including the target(s)
    """
    targets = torch.tensor(get_theta_bus(dataset, obs).real).unsqueeze(dim=2)
    return targets.float()

def get_edge_index_from_ybus(ybus_matrix) -> list:
    """Get all the edge_indices from Ybus matrix

    Parameters
    ----------
    ybus_matrix : _type_
        Ybus matrix as input (NxMxM)
        with N number of observations
        and M number of nodes in the graph

    Returns
    -------
    ``list``
        a list of edge indices
    """
    edge_indices = []
    for ybus in ybus_matrix:
        np.fill_diagonal(ybus, val=0.)
        bus_or, bus_ex = np.where(ybus)
        edge_index = np.column_stack((bus_or, bus_ex)).T
        #edge_index = torch.tensor(edge_index, device=self.device)
        edge_indices.append(edge_index)
    return edge_indices

def get_edge_weights_from_ybus(ybus_matrix, edge_indices) -> list:
    """Get edge weights corresponding to each edge index

    Parameters
    ----------
    ybus_matrix : _type_
        _description_
    edge_indices : _type_
        edge indices returned by the get_edge_index_from_ybus function

    Returns
    -------
    ``list``
        a list of edge weights
    """
    edge_weights = []
    for edge_index, ybus in zip(edge_indices, ybus_matrix):
        edge_weight = []
        for i in range(edge_index.shape[1]):
            edge_weight.append(ybus[edge_index[0][i], edge_index[1][i]])
        #for el in edge_index:
            #edge_weight.append(ybus[el[0], el[1]])
        edge_weight = np.array(edge_weight)
        #edge_weight = torch.tensor(edge_weight, device=self.device)
        edge_weights.append(edge_weight)
    return edge_weights

def get_batches_pyg_tensor_custom(edge_indices,
                           features,
                           targets,
                           batch_size=128,
                           device="cpu",
                           edge_weights=None,
                           ybus=None,
                           line_status=None,
                           topo_vect=None):
    torch_dataset = []
    for i, feature in enumerate(features):
        if edge_weights is not None:
            edge_weight = torch.abs(torch.tensor(edge_weights[i], dtype=torch.float))
        else:
            edge_weight=None
        sample_data = Data(x=feature,
                           y=targets[i],
                           edge_index=torch.tensor(edge_indices[i]),
                           edge_attr=edge_weight,
                           ybus=ybus[i],
                           line_status=line_status[i],
                           topo_vect=topo_vect[i])
        sample_data.to(device)
        torch_dataset.append(sample_data)
    loader = pyg_DataLoader(torch_dataset, batch_size=batch_size)

    return loader

def get_batches_pyg(edge_indices, features, targets, edge_weights=None, batch_size=128, device="cpu"):
    torch_dataset = []
    for i, feature in enumerate(features):
        if edge_weights is not None:
            edge_weight = torch.abs(torch.tensor(edge_weights[i], dtype=torch.float))
        else:
            edge_weight=None
        sample_data = Data(x=feature,
                          y=targets[i],
                          edge_index=torch.tensor(edge_indices[i]),
                          edge_attr=edge_weight)
        sample_data.to(device)
        torch_dataset.append(sample_data)
    loader = pyg_DataLoader(torch_dataset, batch_size=batch_size)

    return loader

def get_active_power_batch(ybus, obs, theta, index):
    """Computes the active power (flows) from thetas (subs) for an index

    Parameters
    ----------
    dataset : _type_
        _description_
    obs : _type_
        _description_
    theta : _type_
        _description_
    index : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    lor_bus, lor_conn = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
    lex_bus, lex_conn = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)
    index_array = np.vstack((np.arange(obs.n_line), lor_bus, lex_bus)).T
    # Create the adjacency matrix (MxN) M: branches and N: Nodes
    A_or = np.zeros((obs.n_line, obs.n_sub))
    A_ex = np.zeros((obs.n_line, obs.n_sub))

    for line in index_array[:,0]:
        if index_array[line,1] != -1:
            A_or[line, index_array[line,1]] = 1
            A_or[line, index_array[line,2]] = -1
            A_ex[line, index_array[line,1]] = -1
            A_ex[line, index_array[line,2]] = 1
    
    # Create the diagonal matrix D (MxM)
    Ybus = ybus[index][:14,:14]
    D = np.zeros((obs.n_line, obs.n_line), dtype=complex)
    for line in index_array[:, 0]:
        bus_from = index_array[line, 1]
        bus_to = index_array[line, 2]
        D[line,line] = Ybus[bus_from, bus_to] * (-1)

    # Create the theta vector ((M-1)x1)
    theta = 1j*((theta[index,1:]*pi)/180)
    p_or = (D.dot(A_or[:,1:])).dot(theta.reshape(-1,1))
    p_ex = (D.dot(A_ex[:,1:])).dot(theta.reshape(-1,1))

    #return p_or, p_ex
    return p_or.imag * 100 , p_ex.imag * 100

def get_all_active_powers_batch(batch, obs, theta_bus):
    """Computes all the active powers for all the observations from theta at bus

    Parameters
    ----------
    dataset : _type_
        _description_
    obs : _type_
        _description_
    theta_bus : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # data_size = len(dataset["p_or"])
    # p_or = np.zeros_like(dataset["p_or"])
    # p_ex = np.zeros_like(dataset["p_ex"])
    batch_size = len(batch.ybus)
    p_or = np.zeros(shape=(batch_size, obs.n_line))
    p_ex = np.zeros(shape=(batch_size, obs.n_line))

    #theta_bus = get_theta_bus(dataset, obs)
    for ind in range(batch_size):
        obs = create_fake_obs_custom(obs, batch.line_status, batch.topo_vect, ind)
        p_or_computed, p_ex_computed = get_active_power_batch(batch.ybus, obs, theta_bus, index=ind)
        p_or[ind, :] = p_or_computed.flatten()
        p_ex[ind, :] = p_ex_computed.flatten()
    
    return p_or, p_ex

def get_active_power(dataset, obs, theta, index):
    """Computes the active power (flows) from thetas (subs) for an index

    Parameters
    ----------
    dataset : _type_
        _description_
    obs : _type_
        _description_
    theta : _type_
        _description_
    index : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    lor_bus, lor_conn = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
    lex_bus, lex_conn = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)
    index_array = np.vstack((np.arange(obs.n_line), lor_bus, lex_bus)).T
    # Create the adjacency matrix (MxN) M: branches and N: Nodes
    A_or = np.zeros((obs.n_line, obs.n_sub))
    A_ex = np.zeros((obs.n_line, obs.n_sub))

    for line in index_array[:,0]:
        if index_array[line,1] != -1:
            A_or[line, index_array[line,1]] = 1
            A_or[line, index_array[line,2]] = -1
            A_ex[line, index_array[line,1]] = -1
            A_ex[line, index_array[line,2]] = 1
    
    # Create the diagonal matrix D (MxM)
    Ybus = dataset["YBus"][index][:obs.n_sub,:obs.n_sub]
    D = np.zeros((obs.n_line, obs.n_line), dtype=complex)
    for line in index_array[:, 0]:
        bus_from = index_array[line, 1]
        bus_to = index_array[line, 2]
        D[line,line] = Ybus[bus_from, bus_to] * (-1)

    # Create the theta vector ((M-1)x1)
    theta = 1j*((theta[index,1:]*pi)/180)
    p_or = (D.dot(A_or[:,1:])).dot(theta.reshape(-1,1))
    p_ex = (D.dot(A_ex[:,1:])).dot(theta.reshape(-1,1))

    #return p_or, p_ex
    return p_or.imag * 100 , p_ex.imag * 100

def get_all_active_powers(dataset, obs, theta_bus):
    """Computes all the active powers for all the observations from theta at bus

    Parameters
    ----------
    dataset : _type_
        _description_
    obs : _type_
        _description_
    theta_bus : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    data = dataset.data
    p_or = np.zeros_like(data["p_or"])
    p_ex = np.zeros_like(data["p_ex"])

    #theta_bus = get_theta_bus(dataset, obs)
    for ind in range(dataset.size):
        obs = create_fake_obs(obs, data, ind)
        p_or_computed, p_ex_computed = get_active_power(data, obs, theta_bus, index=ind)
        p_or[ind, :] = p_or_computed.flatten()
        p_ex[ind, :] = p_ex_computed.flatten()

    return p_or, p_ex

def reconstruct_theta_line(dataset, obs, theta_sub) -> tuple:
    """Reconstruct the voltage angles through lines from thetas on subs

    Parameters
    ----------
    dataset : _type_
        _description_
    obs : _type_
        _description_
    theta_sub : _type_
        _description_

    Returns
    -------
    tuple
        theta_or: voltage angle at the origin side of the power line
        theta_ex: voltage angle at the extrem side of the power line
    """
    data = dataset.data
    theta_or = np.zeros_like(data["theta_or"])
    theta_ex = np.zeros_like(data["theta_ex"])

    for idx in range(dataset.size):
        obs.line_status = data["line_status"][idx]
        obs.topo_vect = data["topo_vect"][idx]
        lor_bus, _ = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
        lex_bus, _ = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)
        index_array = np.vstack((np.arange(obs.n_line), lor_bus, lex_bus)).T

        for line in range(obs.n_line):
            if index_array[line, 1] != -1:
                theta_or[idx][line] = theta_sub[idx][index_array[line, 1]]
                theta_ex[idx][line] = theta_sub[idx][index_array[line, 2]]
    return theta_or, theta_ex
