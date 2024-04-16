""" This module is used to generate powergrid datasets

Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

import os
import warnings
import shutil
import copy
from typing import Union, Callable
import json
from tqdm import tqdm 
import zipfile
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse
from grid2op.Agent import BaseAgent

from . import DataSet
from ..utils import NpEncoder
from ..logger import CustomLogger
from ..physical_simulator import Grid2opSimulator
from ..config import ConfigManager

class DownloadProgressBar(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b = 1, bsize = 1, tsize = None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n) # also sets self.n = b * bsize

def downloadPowergridDataset(path: str, dataset_name: str="lips_case14_sandbox", unzip: bool=True):
    """Download the dataset

    Parameters
    ----------
    path : ``str``
        path to download the dataset

    dataset_name : ``str``, optional
        the dataset name, by default "l2rpn_case14_sandbox"
    
    unzip : ``bool``, optional
        whether to unzip the downloaded file, by default True

    Function
    ------
    Download the powergrid dataset and unzip it in the given path
    available datasets are:
        - lips_case14_sandbox
        - lips_neurips_2020_track1_small
        - lips_idf_2023 
    """
    try:
        base_url = "https://data.lips.irt-systemx.fr/data/"
        url = base_url + dataset_name + ".zip"
        os.makedirs(path, exist_ok = True)
        with DownloadProgressBar(unit = 'B', unit_scale = True, miniters = 1, unit_divisor = 1024, desc = 'Downloading test') as t:
            urlretrieve(url, filename = os.path.join(path, dataset_name + '.zip'), reporthook = t.update_to)

        if unzip:
            print("Extracting " + dataset_name + ".zip at " + path + "...")
            with zipfile.ZipFile(os.path.join(path, dataset_name + '.zip'), 'r') as zipf:
                zipf.extractall(path)

    except ImportError as exc_:
        raise RuntimeError("Impossible to `download` powergrid ") from exc_
        
class PowerGridDataSet(DataSet):
    """Class to manage powergrid data

    This specific DataSet uses grid2op framework to simulate data coming from a powergrid.

    Attributes
    ----------
    name : ``str``, optional
        the dataset name, by default "train"
    attr_names : Union[``tuple``, ``None``], optional
        the attributes list for which data should be generated, by default None
    log_path : Union[``str``, ``None``], optional
        the path where the logs should be stored, by default None

    Todo
    -------
    TODO: to remove all the comments lines corresponding to theta attributes

    Examples
    --------

    .. code-block:: python

        from lips.dataset import PowerGridDataSet

        # create a dataset
        dataset = PowerGridDataSet()
        dataset.generate_data()

    """
    ALL_VARIABLES = ("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                     "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                     "v_or", "v_ex", "theta_or", "theta_ex")

    def __init__(self,
                 config: ConfigManager,
                 name: str="train",
                 # for compatibility with existing code this will be removed in future version
                 # (and serialize directly the output of the simulator)
                 attr_names: Union[tuple, None]=None,
                 log_path: Union[str, None]=None
                 ):
        DataSet.__init__(self, name=name)
        self._nb_divergence = 0
        if attr_names is not None:
            self._attr_names = copy.deepcopy(attr_names)
        else:
            self._attr_names = self.ALL_VARIABLES
        self.size = 0

        self.chronics_info = dict()

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # config
        self.config = config
        # for the sampling
        self._previous = None
        self._order = None

        # normalization parameters
        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_tau = None
        self._size_y = None
        self._sizes_x = None  # dimension of each variable
        self._sizes_tau = None
        self._sizes_y = None  # dimension of each variable
        self._attr_tau = self.config.get_option("attr_tau")
        self._attr_x = self.config.get_option("attr_x")
        self._attr_y = self.config.get_option("attr_y")

        self.env_data = dict()
        self._slack_id = None

        #TODO add a seed for reproducible experiment !

    def download(self, path: str, dataset_name: str="lips_case14_sandbox", unzip: bool=True):
        downloadPowergridDataset(path, dataset_name, unzip)
            

    def generate(self,
                 simulator: Grid2opSimulator,
                 actor: Union[BaseAgent, None],
                 path_out: Union[str, None],
                 nb_samples: int,
                 nb_samples_per_chronic: int=288, # 1 day
                 simulator_seed: Union[None, int]=None,
                 actor_seed: Union[None, int]=None,
                 do_store_physics: bool=False,
                 store_as_sparse: bool=False,
                 is_dc: bool=False):
        """Generate a powergrid dataset

        For this dataset, we use a Grid2opSimulator and a  grid2op Agent to generate data from a powergrid.
        This implementation can also serve as a reference for other implementation of the `generate` function.

        Parameters
        ----------
        simulator : Grid2opSimulator
            In this case, this should be a grid2op environment
        actor : Union[``BaseAgent``, ``None``]
            the actor used to generate the data
        path_out : Union[``str``, ``None``]
            The path where the data will be saved
        nb_samples : ``int``
            Number of rows (examples) in the final dataset
        simulator_seed : Union[``None``, ``int``], optional
            Seed used to set the simulator for reproducible experiments, by default None
        actor_seed : Union[``None``, ``int``], optional
            Seed used to set the actor for reproducible experiments, by default None

        Raises
        ------
        RuntimeError
            Impossible to generate powergird data, Grid2Op is not installed
        RuntimeError
            Impossible to generate negative number of data

        """
        try:
            from grid2op.Agent import DoNothingAgent
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` powergrid data if you don't have "
                               "the grid2Op package installed") from exc_
        self._nb_divergence = 0
        if nb_samples <= 0:
            raise RuntimeError("Impossible to generate a negative number of data.")

        # check that the proper data types are received
        super().generate(simulator, actor, path_out, nb_samples, simulator_seed, actor_seed)

        if actor is None:
            # TODO refactoring this, this is weird here
            actor = DoNothingAgent(simulator._simulator.action_space)

        init_state, _ = simulator.get_state()

        prod_bus, _ = init_state._get_bus_id(init_state.gen_pos_topo_vect, init_state.gen_to_subid)
        self._slack_id = prod_bus[-1] #in the hypothesis that there is only one slack and that it does not change accross observations

        self.data = {}
        for attr_nm in self._attr_names:
            # this part is only temporary, until a viable way to store the complete resulting state is found
            array_ = getattr(init_state, attr_nm)
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)
        
        #to save physical variables if required
        if (do_store_physics):
            n_bus_bars = simulator._simulator.n_sub * 2
            # store the Ybus as sparse matrix, mendatory for large envs as the memory is highly impacted by this matrix
            if store_as_sparse:
                self.__ybus_data = []
                self.__row_indices = []
                self.__col_indices = []
            else:
                self.data["YBus"] = np.zeros((nb_samples, n_bus_bars, n_bus_bars), dtype=np.complex128) #for admittance matrix
            self.data["SBus"] = np.zeros((nb_samples, n_bus_bars), dtype=np.complex128) #for vector of nodal injections
            self.data["PV_nodes"] = np.zeros((nb_samples, n_bus_bars), dtype=bool) #for indices of P-V nodes
            self.data["slack"] = np.zeros((nb_samples, 2), dtype=np.float16) #for indice of slack and active power adjusment from chronic

        ds_size = 0
        pbar = tqdm(total = nb_samples)
        is_LightSimBackend = ("lightsim2grid" in str(simulator._simulator.backend.__class__))
        while ds_size < nb_samples:
            simulator.sample_chronics()
            nb_steps = 0
            while (nb_steps < nb_samples_per_chronic):
                simulator.modify_state(actor)
                current_state, _ = simulator.get_state()
                grid = simulator._simulator.backend._grid
                self._store_obs(ds_size, current_state, grid, do_store_physics, is_LightSimBackend, is_dc, store_as_sparse)
                nb_steps += 1
                ds_size += 1
                pbar.update(1)
                if ds_size >= nb_samples:
                    break
        if do_store_physics & store_as_sparse:
            sparse_matrix = sparse.csr_matrix((self.__ybus_data, (self.__row_indices, self.__col_indices)),
                                              shape=(nb_samples,n_bus_bars*n_bus_bars))
            self.data["YBus"] = sparse_matrix
        pbar.close()
        self.size = nb_samples
        self._init_sample()
        self._infer_sizes()
        self._store_env_data(simulator._simulator)
        self._store_chronics_info(simulator)
        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out, do_save_physics=do_store_physics, store_as_sparse=store_as_sparse)

    def _store_obs(self, current_size, obs, grid, do_store_physics=True, is_lightSimBackend=True, is_dc=False, store_as_sparse=False):
        """store an observation in self.data"""
        for attr_nm in self._attr_names:
            array_ = getattr(obs, attr_nm)
            self.data[attr_nm][current_size, :] = array_

        #Also Store Admittance Matrix YBus and Injection Vector SBus (if backend is LightSim2Grid
        if(do_store_physics):
            self._store_physics(current_size, obs, grid, is_lightSimBackend, is_dc, store_as_sparse)


    def _store_physics(self, current_size: int, obs, grid, is_lightSimBackend: bool, is_dc: bool=False, store_as_sparse: bool=False):
        """Store physical variables related to grid in addition to regular observations

        Parameters
        ----------
        current_size : _type_
            _description_
        obs : _type_
            _description_
        grid : _type_
            _description_
        is_lightSimBackend : bool
            _description_
        is_dc : bool, optional
            _description_, by default False
        """
        if is_lightSimBackend:
            # net = obs._obs_env.backend._grid #be careful, the YBus and Sbus are not well updated in _obs_env. Better to use the true env backend
            nb_bus, unique_bus, bus_or, bus_ex = obs._aux_fun_get_bus()

            # Careful, in computing the number of bus_bars, it could change in grid2op
            n_bus_bars = obs._obs_env.n_sub * 2

            #Init data structures with proper dimensions and types
            admittance_matrix = np.zeros(shape=(n_bus_bars, n_bus_bars), dtype=np.complex128)
            Injection_vect = np.zeros(shape=(n_bus_bars), dtype=np.complex128)
            pv_nodes = np.zeros(shape=(n_bus_bars), dtype=bool)

            #get admittance matrix and injection vector values from LightSim2Grid
            Sbus = grid.get_Sbus()
            if is_dc:
                YBus = grid.get_dcYbus()
            else:
                YBus = grid.get_Ybus()

            #fill data structures with expected dimensions
            admittance_matrix[np.ix_(unique_bus, unique_bus)] = YBus.todense()
            Injection_vect[unique_bus] = Sbus
            pv_nodes[unique_bus[grid.get_pv()]] = True

            #get info on slack
            prod_bus, _ = obs._get_bus_id(obs.gen_pos_topo_vect, obs.gen_to_subid)
            node_slack_id = prod_bus[-1]
            index_gens_slack = (prod_bus==node_slack_id)
            adjusted_prod_slack = obs.gen_p[index_gens_slack].sum() - Sbus[node_slack_id].real #after_powerflow - in_chronics

            if store_as_sparse:
                array_2d = admittance_matrix.reshape(1,-1)
                row_index, col_index = np.nonzero(array_2d)
                data = array_2d[row_index, col_index]
                self.__row_indices.extend(row_index + current_size)
                self.__col_indices.extend(col_index)
                self.__ybus_data.extend(data)
            else:
                self.data["YBus"][current_size, :] = admittance_matrix
            self.data["SBus"][current_size, :] = Injection_vect
            self.data["PV_nodes"][current_size, :] = pv_nodes
            self.data["slack"][current_size, :] = np.array([node_slack_id,adjusted_prod_slack], dtype=np.float16)
        else:
            self.logger.warning("Cannot store physics data if not using LightSim2Grid backend")

    def _store_chronics_info(self, simulator):
        """store the chronics info"""
        chronics_features = ["chronics_id", "chronics_name", "chronics_name_unique", "time_stamps"]
        for attr_nm in chronics_features:
            array_ = getattr(simulator, attr_nm)
            self.chronics_info[attr_nm] = array_


    def _store_env_data(self, env):
        """store the environment data in self.env_data"""
        self.env_data["n_line"] = env.n_line
        self.env_data["n_sub"] = env.n_sub
        self.env_data["n_gen"] = env.n_gen
        self.env_data["n_load"] = env.n_load
        self.env_data["sub_info"] = env.sub_info
        self.env_data["_size_x"] = self._size_x
        self.env_data["_size_tau"] = self._size_tau
        self.env_data["_size_y"] = self._size_y
        self.env_data["_sizes_x"] = self._sizes_x
        self.env_data["_sizes_tau"] = self._sizes_tau
        self.env_data["_sizes_y"] = self._sizes_y
        self.env_data["_attr_x"] = self._attr_x
        self.env_data["_attr_tau"] = self._attr_tau
        self.env_data["_attr_y"] = self._attr_y

    def _save_chronics_info(self, path_out: str):
        """save chronics information as npz"""
        chronics_path = os.path.join(path_out, "chronics")
        os.mkdir(chronics_path)
        for attr_nm, array_ in self.chronics_info.items():
            if attr_nm == "time_stamps":
                array_ = np.asanyarray(array_, dtype=object)
            np.savez_compressed(f"{os.path.join(chronics_path, attr_nm)}.npz", data=array_)

    def _save_internal_data(self, path_out:str,do_save_physics=True, store_as_sparse: bool=False):
        """save the self.data in a proper format

        Parameters
        ----------
        path_out : ``str``
            path to save the data

        """
        full_path_out = os.path.join(os.path.abspath(path_out), self.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))
            # TODO logger
            #print(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")
            self.logger.info(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")

        if os.path.exists(full_path_out):
            # deleting previous saved data
            # TODO logger
            #print(f"Deleting previous run at {full_path_out}")
            self.logger.warning(f"Deleting previous run at {full_path_out}")
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)
        # TODO logger
        #print(f"Creating the path {full_path_out} to store the dataset name {self.name}")
        self.logger.info(f"Creating the path {full_path_out} to store the dataset name {self.name}")

        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        for attr_nm in self._attr_names:
            np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data=self.data[attr_nm])
        if(do_save_physics):
            if("YBus" in self.data.keys()):
                if store_as_sparse:
                    sparse.save_npz(os.path.join(full_path_out, "YBus")+".npz", matrix=self.data["YBus"])
                else:
                    np.savez_compressed(os.path.join(full_path_out, "YBus")+".npz", data=self.data["YBus"])
            if("SBus" in self.data.keys()):
                np.savez_compressed(os.path.join(full_path_out, "SBus") + ".npz", data=self.data["SBus"])
            if("PV_nodes" in self.data.keys()):
                np.savez_compressed(os.path.join(full_path_out, "PV_nodes") + ".npz", data=self.data["PV_nodes"])
            if("slack" in self.data.keys()):
                np.savez_compressed(os.path.join(full_path_out, "slack") + ".npz", data=self.data["slack"])
                
        
        self._save_chronics_info(full_path_out)
        # save static_data
        with open(os.path.join(full_path_out ,"metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=self.env_data, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def load(self, path:str, load_ybus_as_sparse: bool=False):
        """load the dataset from a path

        Parameters
        ----------
        path : ``str``
            path from which the data should be loaded

        Raises
        ------
        RuntimeError
            Path cannot be found
        RuntimeError
            Not a valid directory
        RuntimeError
            There is no data to load
        RuntimeError
            Impossible to load the data

        """
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")
        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            if not os.path.exists(path_this_array):
                raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                                   f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

        if self.data is not None:
            warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
        self.data = {}
        self.size = None
        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        if self.config.get_option("attr_physics"):
            attr_names = self._attr_names + self.config.get_option("attr_physics")
        else:
            attr_names = self._attr_names

        for attr_nm in attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            if (attr_nm == "YBus") and (load_ybus_as_sparse):
                self.data[attr_nm] = sparse.load_npz(path_this_array)
            else:
                self.data[attr_nm] = np.load(path_this_array)["data"]

        self.size = self.data[attr_nm].shape[0]

        self._init_sample()
        self._infer_sizes()
        self._load_env_data(full_path)
        self._load_chronics_info(full_path)

    def _load_env_data(self, path:str):
        """load the environment data from a path

        Parameters
        ----------
        path : ``str``
            path from which the data should be loaded

        Raises
        ------
        RuntimeError
            Path cannot be found
        RuntimeError
            Not a valid directory
        RuntimeError
            There is no data to load
        RuntimeError
            Impossible to load the data

        """
        if not os.path.exists(os.path.join(path, "metadata.json")):
            raise RuntimeError(f"There is no metadata saved in {path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")

        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
            self.env_data = json.load(fp=f)

    def _load_chronics_info(self, path: str):
        """save chronics information as npz"""
        chronics_features = ["chronics_id", "chronics_name", "chronics_name_unique", "time_stamps"]
        chronics_path = os.path.join(path, "chronics")
        if os.path.exists(chronics_path):
            for attr_nm in chronics_features:
                path_this_array = f"{os.path.join(chronics_path, attr_nm)}.npz"
                self.chronics_info[attr_nm] = np.load(path_this_array, allow_pickle=True)["data"]

    def _init_sample(self):
        """initialize the sample

        Init the sample
        """
        self._previous = 0
        self._order = np.arange(self.size)
        np.random.shuffle(self._order)

    def sample(self, nb_sample: int, sampler: Callable=None):
        """Sampling from the dataset

        For now, this sampling method will sample uniformly at random from the dataset.

        There is a guarantee: if you generate `sef.size` consecutive data with this method
        for example `batch1 = dataset.sample(data.size / 2)` then `batch2 = dataset.sample(data.size / 2)`
        in this case batch1 and batch2 will count different example of the dataset and the union batch1 and batch2
        will make the entire dataset [NB the above is true if the dataset has just been created, or if batch1
        comes from the first batch issued from this dataset.]

        Parameters
        ----------
        nb_sample : ``int``
            Number of sample to retrieve from the dataset.

        sampler : Callable
            currently unused

        Returns
        -------
        dict
            A batch of data, either for training or for testing.

        Raises
        ------
        RuntimeError
            Impossible to require a negative number of data
        RuntimeError
            Impossible to require more than the size of the dataset

        """
        if nb_sample < 0:
            raise RuntimeError("Impossible to require a negative number of data.")
        if nb_sample > self.size:
            raise RuntimeError("Impossible to require more than the size of the dataset")

        res = {}
        if nb_sample + self._previous < self.size:
            # i just sample the next batch of data
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el] = self.data[el][self._order[self._previous:(self._previous+nb_sample)], :]
            self._previous += nb_sample
        else:
            this_sz = self.size - self._previous
            # init the results
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)
            # fill with the remaining of the data
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el][:this_sz] = self.data[el][self._order[self._previous:], :]

            # sample another order to see the data
            self._init_sample()
            # fill with the remaining of the data
            self._previous = nb_sample - this_sz
            #for el in (*self._attr_names, *self._theta_attr_names):
            for el in self._attr_names:
                res[el][this_sz:] = self.data[el][self._order[:self._previous], :]
        return res

    def get_data(self, index: tuple) -> dict:
        """This function returns the data in the data that match the index `index`

        Parameters
        ----------
        index : ``tuple``
            A list of integer

        Returns
        -------
        dict
            a dictionary of key (variable name) value (variable value)

        """
        super().get_data(index)  # check that everything is legit

        # make sure the index are numpy array
        if isinstance(index, list):
            index = np.array(index, dtype=int)
        elif isinstance(index, int):
            index = np.array([index], dtype=int)

        # init the results
        res = {}
        nb_sample = index.size
        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)

        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el][:] = self.data[el][index, :]

        return res

    def _infer_sizes(self):
        #data = copy.deepcopy(self.data)
        self._sizes_x = np.array([self.data[el].shape[1] for el in self._attr_x], dtype=int)
        self._size_x = np.sum(self._sizes_x)
        self._sizes_y = np.array([self.data[el].shape[1] for el in self._attr_y], dtype=int)
        self._size_y = np.sum(self._sizes_y)
        if self._attr_tau is not None:
            self._sizes_tau = np.array([self.data[el].shape[1] for el in self._attr_tau], dtype=int)
            self._size_tau = np.sum(self._sizes_tau)

    def get_sizes(self, attr_x: Union[tuple, None]=None, attr_tau: Union[tuple, None]=None, attr_y: Union[tuple, None]=None):
        """Get the sizes of the dataset

        Returns
        -------
        tuple
            A tuple of size (nb_sample, size_x, size_tau, size_y)

        """
        if attr_x is not None:
            sizes_x = np.array([self.data[el].shape[1] for el in attr_x], dtype=int)
            size_x = np.sum(sizes_x)
        else:
            size_x = self._size_x

        if attr_y is not None:
            sizes_y = np.array([self.data[el].shape[1] for el in attr_y], dtype=int)
            size_y = np.sum(sizes_y)
        else:
            size_y = self._size_y

        if attr_tau is not None:
            sizes_tau = np.array([self.data[el].shape[1] for el in attr_tau], dtype=int)
            size_tau = np.sum(sizes_tau)
        else:
            size_tau = self._size_tau
            #return size_x, size_y
        
        return size_x, size_tau, size_y

    def extract_data(self, concat: bool=True) -> tuple:
        """extract the x and y data from the dataset

        Parameters
        ----------
        concat : ``bool``
            If True, the data will be concatenated in a single array.
        Returns
        -------
        tuple
            extracted inputs and outputs
        """
        # init the sizes and everything
        data = copy.deepcopy(self.data)

        if concat:
            if self._attr_tau is not None:
                attr_x = self._attr_x + self._attr_tau
            else:
                attr_x = self._attr_x
            extract_x = np.concatenate([data[el].astype(np.float32) for el in attr_x], axis=1)
            extract_y = np.concatenate([data[el].astype(np.float32) for el in self._attr_y], axis=1)
            return extract_x, extract_y
        else:
            extract_x = [data[el].astype(np.float32) for el in self._attr_x]
            extract_y = [data[el].astype(np.float32) for el in self._attr_y]
            if self._attr_tau is not None:
                extract_tau = [data[el].astype(np.float32) for el in self._attr_tau]
                return (extract_x, extract_tau), extract_y
            else:
                return extract_x, extract_y

    def reconstruct_output(self, data: "np.ndarray") -> dict:
        """It reconstruct the data from the extracted data

        Parameters
        ----------
        data : ``np.ndarray``
            the array that should be reconstruted

        Returns
        -------
        dict
            the reconstructed data with variable names in a dictionary
        """
        predictions = {}
        prev_ = 0
        for var_id, this_var_size in enumerate(self._sizes_y):
            attr_nm = self._attr_y[var_id]
            predictions[attr_nm] = data[:, prev_:(prev_ + this_var_size)]
            prev_ += this_var_size
        return predictions
