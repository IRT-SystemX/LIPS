""" This module is used to generate pneumatic datasets

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
import numpy as np
import shutil
import csv

import copy
from typing import Union
from tqdm import tqdm 

from lips.dataset.dataSet import DataSet
from lips.logger import CustomLogger
from lips.physical_simulator.GetfemSimulator.GetfemSimulatorBridge import GetfemInterpolationOnSupport,InterpolationOnCloudPoints
from lips.config.configmanager import ConfigManager
from lips.physical_simulator.getfemSimulator import GetfemSimulator
from lips.dataset.sampler import LHSSampler

def domain_2D_grid_generator(origin:tuple, lenghts:tuple, sizes:tuple) -> np.ndarray:
    """Generate 2D grid nodes

        Parameters
        ----------
        origin : tuple
            2D grid origin coordinates
        lenghts : tuple
            grid lenght along x-axis/y-axis        
        sizes : tuple
            grid number of nodes along x-axis/y-axis 
        
        Returns
        -------
        np.ndarray
            nodes coordinates within the 2D grid generated
    """
    origin_x,origin_y=origin
    lenght_x,lenght_y=lenghts
    nb_line,nb_column=sizes
    coord_x,coord_y=np.meshgrid(np.arange(origin_x,origin_x+lenght_x,lenght_x/nb_line),np.arange(origin_y,origin_y+lenght_y,lenght_y/nb_column))
    grid_support_points = np.vstack(list(zip(coord_x.ravel(), coord_y.ravel()))).transpose()
    return grid_support_points

class DataSetInterpolatorOnGrid():
    """Used to interpolate dataset on a grid

        Parameters
        ----------
        name : str
            instance name
        simulator : GetfemSimulator
            physical simulator       
        dataset : DataSet
            dataset 
        grid_support : dict
            grid support features
        
    """
    def __init__(self,
                 name:str,
                 simulator:GetfemSimulator,
                 dataset:DataSet,
                 grid_support):
        self.name = name
        self.simulator = simulator
        self.dataset = dataset
        self.grid_support = grid_support
        self.grid_support_points = np.array([])
        self.interpolated_dataset = dict()
        self.distributed_inputs_on_grid = dict()

    def generate(self, dofnum_by_field:dict, path_out:Union[None, str]=None):
        """Generate an interpolated dataset from a dataset
        Function to generate interpolated data for a given dataset, using the physical simulator.

        We expect that the results are saved in `path_out/` if they are stored
        in plain text.

        Parameters
        ----------
        dofnum_by_field: number of degrees of freedom by field
            A dict that provides the mapping between a field name and the number of degrees of freedom per nodes associated to it.

        """
        self.generate_interpolation_fields(dofnum_by_field)
        self.distribute_data_on_grid()

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)
    
    def generate_interpolation_fields(self, dofnum_by_field:dict):
        """Generate interpolated fields from a dataset
        Function to generate fields from a dataset, using the physical simulator.

        Parameters
        ----------
        dofnum_by_field: number of degrees of freedom by field
            A dict that provides the mapping between a field name and the number of degrees of freedom per nodes associated to it.

        """
        self._init_interpolation_fields(dofnum_by_field)

        grid_shape=self.grid_support["sizes"]
        self.grid_support_points=domain_2D_grid_generator(origin=self.grid_support["origin"],
                                                       lenghts=self.grid_support["lenghts"],
                                                       sizes=grid_shape)

        for data_index in range(len(self.dataset)):
            data_solver_obs=self.dataset.get_data(data_index)
            for field_name,dofnum in dofnum_by_field.items():
                single_field=np.zeros((dofnum,grid_shape[0],grid_shape[1]))
                original_field=data_solver_obs[field_name]
                interpolated_field=GetfemInterpolationOnSupport(self.simulator,original_field,self.grid_support_points)
                for dof_index in range(dofnum):
                    intermediate_field=interpolated_field[dof_index::dofnum]
                    single_field[dof_index]=intermediate_field.reshape((grid_shape[0],grid_shape[1]))
                self.interpolated_dataset[field_name][data_index]=single_field


    def distribute_data_on_grid(self):
        """Distribute data on the grid
        Function to distribute data on the grid, replicate inputs on each nodes
        """
        samples=self.dataset._inputs
        fieldNum=[len(samples[0].keys()) for sample in samples]
        if fieldNum.count(fieldNum[0]) != len(fieldNum):
            raise RuntimeError("Samples do not have the same input parameters")
        value_by_input_attrib = {attribName: np.array([sample[attribName] for sample in samples]) for attribName in samples[0]}

        nx,ny=self.grid_support["sizes"]
        for attrib_name,data in value_by_input_attrib.items():
            data = np.repeat(data[:, np.newaxis], nx*ny, axis=1)
            data = np.reshape(data,(data.shape[0],nx,ny))
            self.distributed_inputs_on_grid[attrib_name]=data

    def _init_interpolation_fields(self, dofnum_by_field:dict):
        """Initialize interpolated field
        Function to initialize interpolated field

        Parameters
        ----------
        dofnum_by_field: number of degrees of freedom by field
            A dict that provides the mapping between a field name and the number of degrees of freedom per nodes associated to it.

        """
        grid_shape=self.grid_support["sizes"]
        for field_name,dof_per_nodes in dofnum_by_field.items():
            self.interpolated_dataset[field_name]=np.zeros((len(self.dataset),dof_per_nodes,grid_shape[0],grid_shape[1]))

    def _save_internal_data(self, path_out:str):
        """Save the internal data in a proper format

        Parameters
        ----------
        path_out: output path
            A str to indicate where to save the data.
        """
        full_path_out = os.path.join(os.path.abspath(path_out), self.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))

        if os.path.exists(full_path_out):
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)

        field_name="GridPoints"
        np.savez_compressed(f"{os.path.join(full_path_out, field_name)}.npz", data=self.grid_support_points)

        for field_name,data in self.interpolated_dataset.items():
            np.savez_compressed(f"{os.path.join(full_path_out, field_name)}Interpolated.npz", data=data)

        for attrib_name,data in self.distributed_inputs_on_grid.items():
            np.savez_compressed(f"{os.path.join(full_path_out, attrib_name)}.npz", data=data)

    def load(self, path:str):
        """Load the internal data

        Parameters
        ----------
        path: input path
            A str to indicate where to load the data from.
        """
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")

        onlyfiles = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
        onlynames=[file.split('.')[0] for file in onlyfiles]

        attr_nm="GridPoints"
        path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
        self.grid_support_points=np.load(path_this_array)["data"]

        interpolated_nm=[name for name in onlynames if "Interpolated" in name]
        for attr_nm in interpolated_nm:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            new_attr_nm=attr_nm.replace('Interpolated','')
            self.interpolated_dataset[new_attr_nm]=np.load(path_this_array)["data"]

        remaining_nm=[name for name in onlynames if "Interpolated" not in name and "GridPoints" not in name]
        for attr_nm in remaining_nm:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.distributed_inputs_on_grid[attr_nm] = np.load(path_this_array)["data"]

    def load_from_data(self,
                       grid_support_points:np.ndarray,
                       interpolated_dataset:np.ndarray,
                       distributed_inputs_on_grid:np.ndarray):
        """Load the internal data from external data

        Parameters
        ----------
        grid_support_points: grid support points
            np.ndarray
        interpolated_dataset: interpolated dataset
            np.ndarray
        distributed_inputs_on_grid: distributed inputs on grid
            np.ndarray
        """
        self.grid_support_points=grid_support_points
        self.interpolated_dataset=interpolated_dataset
        self.distributed_inputs_on_grid=distributed_inputs_on_grid


class DataSetInterpolatorOnMesh():
    """Used to interpolate dataset on a mesh

        Parameters
        ----------
        name : str
            instance name
        simulator : GetfemSimulator
            physical simulator       
        dataset : DataSet
            original dataset to interpolate
    """
    def __init__(self,
                 name:str,
                 simulator:GetfemSimulator,
                 dataset:DataSet):
        self.name=name
        self.simulator=simulator
        self.dataset=dataset
        self.interpolated_dataset = dict()
        self.accumulated_data_from_grid = dict()

    def generate(self, field_names:list, path_out:Union[None, str]=None):
        """Generate an interpolated dataset on a mesh from a dataset
        Function to generate interpolated data for a given dataset on a mesh.

        We expect that the results are saved in `path_out/`

        Parameters
        ----------
        field_names: list of field names to be interpolated
            A list of named fields

        """        
        self.generate_projection_fields_on_mesh(field_names)
        self.accumulate_data_from_grid()

        if path_out is not None:
            self._save_internal_data(path_out)

    def generate_projection_fields_on_mesh(self, field_names:list):
        """Generate projection fields on mesh from a dataset
        Function to generate interpolated field on a mesh.

        Parameters
        ----------
        field_names: list of field names to be interpolated
            A list of named fields

        """  
        self._init_projection_fields(field_names)

        for data_index in range(self.dataset.interpolated_dataset[field_names[0]].shape[0]):
            for field_name in field_names:
                grid_field=self.dataset.interpolated_dataset[field_name][data_index]
                grid_field=np.transpose(grid_field.reshape(grid_field.shape[0],-1))
                field_support=np.transpose(self.dataset.grid_support_points)

                #Clean true zeros
                exterior_points_rows = np.where(grid_field[:,0] == 0.0) and np.where(grid_field[:,1] == 0.0)
                interpolated_interior_sol = np.delete(grid_field, exterior_points_rows, axis=0)
                interpolated_interior_coords=np.delete(field_support, exterior_points_rows, axis=0)
                
                interpolated_field=InterpolationOnCloudPoints(field_support=interpolated_interior_coords,fieldValue=interpolated_interior_sol,phyProblem=self.simulator)
                interleave_interpolated_field=np.empty((interpolated_field.shape[0]*interpolated_field.shape[1],))
                for dof in range(interpolated_field.shape[1]):
                    interleave_interpolated_field[dof::interpolated_field.shape[1]]=interpolated_field[:,dof]
                self.interpolated_dataset[field_name][data_index]=interleave_interpolated_field

    def accumulate_data_from_grid(self):
        """Accumulate data on the grid
        Function to merge input data from all the nodes
        """
        grid_inputs=self.dataset.distributed_inputs_on_grid
        inputs_separated = [dict(zip(grid_inputs,t)) for t in zip(*grid_inputs.values())]
        accumulated_data_from_grid=[None]*len(inputs_separated)
        for obs_id,obs_input in enumerate(inputs_separated):
            obs_input={key:np.mean(value) for key,value in obs_input.items()}
            accumulated_data_from_grid[obs_id]=obs_input
        self.accumulated_data_from_grid={key: [single_data[key] for single_data in accumulated_data_from_grid] for key in accumulated_data_from_grid[0]}

    def _init_projection_fields(self, field_names:list):
        """Initialize projection fields
        Function to initialize projection field

        Parameters
        ----------
        field_names: list of field names to be interpolated
            A list of named fields

        """ 
        nb_samples=self.dataset.interpolated_dataset[field_names[0]].shape[0]
        for field_name in field_names:
            array_ = self.simulator.get_variable_value(field_name=field_name)
            self.interpolated_dataset[field_name] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)

    def _save_internal_data(self, path_out:str):
        """Save the internal data in a proper format

        Parameters
        ----------
        path_out: output path
            A str to indicate where to save the data.
        """
        full_path_out = os.path.join(os.path.abspath(path_out), self.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))

        if os.path.exists(full_path_out):
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)

        for field_name,data in self.interpolated_dataset.items():
            new_field_name=field_name.replace('Interpolated','')
            np.savez_compressed(f"{os.path.join(full_path_out, new_field_name)}.npz", data=data)

        for attrib_name,data in self.accumulated_data_from_grid.items():
            np.savez_compressed(f"{os.path.join(full_path_out, attrib_name)}.npz", data=data)

    def load(self, path:str):
        """Load the internal data

        Parameters
        ----------
        path: input path
            A str to indicate where to load the data from.
        """
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")

        interpolated_nm=self.dataset.interpolated_dataset.keys()
        for attr_nm in interpolated_nm:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.interpolated_dataset[attr_nm]=np.load(path_this_array)["data"]

        remaining_nm=self.dataset.distributed_inputs_on_grid.keys()
        for attr_nm in remaining_nm:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.accumulated_data_from_grid[attr_nm] = np.load(path_this_array)["data"]

class WheelDataSet(DataSet):
    """Base class for pneumatic datasets
    This class represent a single dataset dedicated to the pneumatic usecase, that comes from a database.

    It also offers the possibility to generate data. The data generation come from a simulator
    that will be called when generating the dataset.

    This is the base class of all DataSet in LIPS repository

    Parameters
    ----------
    name: str
        the name of the dataset
    attr_names: tuple
        collection of attributes specific to the dataset
    config: ConfigManager
        instance of configuration
    log_path: str
        path to write the log file
    """
    def __init__(self,
                 config: ConfigManager,
                 name="train",
                 attr_names=("disp",),
                 log_path: Union[str, None]=None,
                 **kwargs):
        super(WheelDataSet,self).__init__(name=name)
        self._attr_names = copy.deepcopy(attr_names)
        self.size = 0
        self._inputs = []

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        self.config = config

        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_y = None
        self._sizes_x = None  # dimension of each variable
        self._sizes_y = None  # dimension of each variable
        self._attr_x = kwargs["attr_x"] if "attr_x" in kwargs.keys() else self.config.get_option("attr_x")
        self._attr_y = kwargs["attr_y"] if "attr_y" in kwargs.keys() else self.config.get_option("attr_y")

    def _infer_sizes(self):
        """Infer the data sizes"""
        data = copy.deepcopy(self.data)
        attrs_x=np.array([np.expand_dims(data[el], axis=1) for el in self._attr_x], dtype=int)
        self._sizes_x = np.array([attr_x.shape[1] for attr_x in attrs_x], dtype=int)
        self._size_x = np.sum(self._sizes_x)

        self._sizes_y = np.array([data[el].shape[1] for el in self._attr_y], dtype=int)
        self._size_y = np.sum(self._sizes_y)

    def _init_store_data(self, simulator:GetfemSimulator, nb_samples:int):
        """Initialize the data (to be stored) sizes"""
        simulator.build_model()
        self.data=dict()
        for attr_nm in self._attr_names:
            array_ = simulator.get_variable_value(field_name=attr_nm)
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)

    def get_sizes(self):
        """Get the sizes of the dataset

        Returns
        -------
        tuple
            A tuple of size (nb_sample, size_x, size_y)

        """
        return self._size_x, self._size_y


    def load(self, path:str):
        """Load the internal data

        Parameters
        ----------
        path: input path
            A str to indicate where to load the data from.
        """
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")

        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            if not os.path.exists(path_this_array):
                raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                                   f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

        if self.data is not None:
            warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
        self.data = {}
        self.size = None

        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.data[attr_nm] = np.load(path_this_array)["data"]
            self.size = self.data[attr_nm].shape[0]

        inputs = {attr_x:self.data[attr_x] for attr_x in self._attr_x}
        self._inputs = [dict(zip(inputs,t)) for t in zip(*inputs.values())]

        self._infer_sizes()

    def load_from_data(self, data:dict):
        """Load the internal data from external data

        Parameters
        ----------
        data: mapping between the data field names and values 
            dict
        """
        self.data = {}
        self.size = None

        for attr_nm in self._attr_names:
            self.data[attr_nm] = data[attr_nm]
            self.size = self.data[attr_nm].shape[0]

        inputs = {attr_x:self.data[attr_x] for attr_x in self._attr_x}
        self._inputs = [dict(zip(inputs,t)) for t in zip(*inputs.values())]

        self._infer_sizes()

    def get_data(self, index:Union[int, list]):
        """
        This function returns the data in the data that match the index `index`

        Parameters
        ----------
        index:
            A list of integer

        Returns
        -------

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
        for el in self._attr_names:
            if len(self.data[el].shape)==1:
                self.data[el]=np.expand_dims(self.data[el], axis=1)
            res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)


        for el in self._attr_names:
            res[el][:] = self.data[el][index, :]

        return res

    def _save_internal_data(self, path_out:str):
        """Save the internal data in a proper format

        Parameters
        ----------
        path_out: output path
            A str to indicate where to save the data.
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

        for attr_nm in self._attr_names:
            np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data=self.data[attr_nm])


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
        extract_x = [data[el].astype(np.float32) for el in self._attr_x]
        extract_y = [data[el].astype(np.float32) for el in self._attr_y]

        if concat:
            if len(extract_x[0].shape)==1:
                extract_x = [single_x.reshape((single_x.shape[0],1)) for single_x in extract_x]
            extract_x = np.concatenate(extract_x, axis=1)
            extract_y = np.concatenate(extract_y, axis=1)
        return extract_x, extract_y

    def __getitem__(self, item:int)->tuple:
        currentInput = {inputName:self.data[inputName][item] for inputName in self._attr_x}
        currentOutput = {outputName:self.data[outputName][item] for outputName in self._attr_y}
        return currentInput,currentOutput

class QuasiStaticWheelDataSet(WheelDataSet):
    """
    This specific DataSet uses Getfem framework to simulate data arising from a rolling wheel problem in a quasi-static configuration.

    This is the derived class of WheelDataSet

    Parameters
    ----------
    name: str
        the name of the dataset
    attr_names: tuple
        collection of attributes specific to the dataset
    config: ConfigManager
        instance of configuration
    log_path: str
        path to write the log file
    """

    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 config: Union[ConfigManager, None]=None,
                 log_path: Union[str, None]=None,
                 **kwargs):
        super(QuasiStaticWheelDataSet,self).__init__(name=name,attr_names=attr_names,config=config,log_path=log_path,**kwargs)

    def generate(self,
                 simulator: GetfemSimulator,
                 path_out: Union[str, None]= None):
        """Generate a pneumatic dataset in a quasi-static configuration

        For this dataset, we use a Getfem simulator to generate data from the pneumatic.

        Parameters
        ----------
        simulator : GetfemSimulator
            In this case, this should be a GetfemSimulator instance
        path_out : Union[``str``, ``None``]
            The path where the data will be saved

        Raises
        ------
        RuntimeError
            Impossible to generate pneumatic data, getfem library is not installed

        """
        try:
            import getfem
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` a wheel dateset if you don't have "
                               "the getfem package installed") from exc_

        transientParams=getattr(simulator._simulator,"transientParams")
        nb_samples=int(transientParams["time"]//transientParams["timeStep"]) + 1
        self._init_store_data(simulator=simulator,nb_samples=nb_samples)
        solverState=simulator.run_problem()
            
        self._store_obs(obs=simulator)

        timesteps=getattr(simulator._simulator,"timeSteps")
        self.data["timeSteps"]=timesteps
        self._infer_sizes()

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)
            attrib_name="timeSteps"
            full_path_out = os.path.join(os.path.abspath(path_out), self.name)
            np.savez_compressed(f"{os.path.join(full_path_out, attrib_name)}.npz", data=timesteps)

    def _store_obs(self, obs:GetfemSimulator):
        """store an observation in internal data

        Parameters
        ----------
        simulator : GetfemSimulator
            In this case, this should be a GetfemSimulator instance
        """
        for attr_nm in self._attr_names:
            array_ = obs.get_solution(field_name=attr_nm)
            self.data[attr_nm] = np.array(array_)

class SamplerStaticWheelDataSet(WheelDataSet):
    """
    This specific DataSet uses Getfem framework to simulate data arising from a rolling wheel problem.

    Parameters
    ----------
    name: str
        the name of the dataset
    attr_names: tuple
        collection of attributes specific to the dataset
    config: ConfigManager
        instance of configuration
    log_path: str
        path to write the log file
    """

    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 config: Union[ConfigManager, None]=None,
                 log_path: Union[str, None]=None,
                 **kwargs
                 ):
        super(SamplerStaticWheelDataSet,self).__init__(name=name,attr_names=attr_names,config=config,log_path=log_path,**kwargs)

    def generate(self,
                 simulator: GetfemSimulator,
                 actor:LHSSampler,
                 nb_samples: int,
                 path_out: Union[str, None]= None,
                 actor_seed: Union[None, int] = None):
        """Generate a pneumatic dataset in a quasi-static configuration

        For this dataset, we use a Getfem simulator to generate data from the pneumatic.

        Parameters
        ----------
        simulator : GetfemSimulator
            In this case, this should be a GetfemSimulator instance
        actor: LHSSampler
            Sampler method in this case
        nb_sample : int
            Number of samples (example) in the dataset.
        path_out : Union[str, None], optional
            The path where the data will be saved (if given)
        actor_seed : Union[None, int], optional
            Seed used to set the actor for reproducible experiments, by default None

        Raises
        ------
        RuntimeError
            Impossible to generate pneumatic data, getfem library is not installed
        RuntimeError
            Impossible to require a negative number of data
        """

        try:
            import getfem
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` a wheel dateset  if you don't have "
                               "the getfem package installed") from exc_
        if nb_samples <= 0:
            raise RuntimeError("Impossible to generate a negative number of data.")

        self._inputs=actor.generate_samples(nb_samples=nb_samples,sampler_seed=actor_seed)
        self._init_store_data(simulator=simulator,nb_samples=nb_samples)

        for current_size,sample in enumerate(tqdm(self._inputs, desc=self.name)):
            simulator=type(simulator)(simulator_instance=simulator)
            simulator.modify_state(state=sample)
            simulator.build_model()
            solverState=simulator.run_problem()
            
            self._store_obs(current_size=current_size,obs=simulator)
        self.size = nb_samples
        actor_attrib=actor.get_attributes_as_data()
        self.data={**self.data, **actor_attrib}
        self._infer_sizes()

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)
            full_path_out = os.path.join(os.path.abspath(path_out), self.name)
            actor.save(path_out=full_path_out)


    def _store_obs(self, current_size: int, obs:GetfemSimulator):
        """store an observation in self.data

        Parameters
        ----------
        current_size : int
            data specific index
        obs : GetfemSimulator
            In this case, this should be a GetfemSimulator instance

        """
        for attr_nm in self._attr_names:
            array_ = obs.get_solution(field_name=attr_nm)
            self.data[attr_nm][current_size, :] = array_


#Check integrities

import math
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
from lips.config import ConfigManager

def check_static_samples_generation(configFilePath):
    physical_domain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.,15.),
        "meshSize":1
    }

    physical_properties={
        "problem_type":"StaticMechanicalStandard",
        "materials":[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]],
        "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0}] ],
        "dirichlet":[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    }
    training_simulator=GetfemSimulator(physical_domain=physical_domain,physical_properties=physical_properties)

    trainingInput={
              "young":(75.0,85.0),
              "poisson":(0.38,0.44),
              "fricCoeff":(0.5,0.8)
              }

    training_actor=LHSSampler(space_params=trainingInput)

    attr_names=(PFN.displacement,PFN.contactMultiplier)

    staticWheelDataSet=SamplerStaticWheelDataSet("train",attr_names=attr_names,attr_x= ("young","poisson","fricCoeff"),attr_y= ("disp",))
    staticWheelDataSet.generate(simulator=training_simulator,
                                    actor=training_actor,
                                    path_out="WheelDir",
                                    nb_samples=5,
                                    actor_seed=42
                                    )
    # print(staticWheelDataSet.get_data(index=0))
    # print(staticWheelDataSet.data)

    #Interpolation on grid
    grid_support={"origin":(-16.0,0.0),"lenghts":(32.0,32.0),"sizes":(16,16)}
    myTransformer=DataSetInterpolatorOnGrid(name="train",
                                            simulator=training_simulator,
                                            dataset=staticWheelDataSet,
                                            grid_support=grid_support)
    dofnum_by_field={PFN.displacement:2}
    myTransformer.generate(dofnum_by_field=dofnum_by_field,path_out="wheel_interpolated")

def check_quasi_static_generation(configFilePath):
    physical_domain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.,15.),
        "meshSize":1
    }

    dt = 10e-4
    physical_properties={
        "problem_type":"QuasiStaticMechanicalRolling",
        "materials":[["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1} ]],
        "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ],
        "rolling":["HOLE_BOUND",{"type" : "DIS_Rolling", "theta_Rolling":150., 'd': 1.}],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.6}] ],
        "transientParams":{"time": 3*dt, "timeStep": dt}
    }

    training_simulator=GetfemSimulator(physical_domain=physical_domain,physical_properties=physical_properties)
    attr_names=(PFN.displacement,PFN.contactMultiplier)

    quasiStaticWheelDataSet=QuasiStaticWheelDataSet("train",attr_names=attr_names,attr_x = ("timeSteps",),attr_y = ("disp","contactMult"))
    quasiStaticWheelDataSet.generate(simulator=training_simulator,
                                    path_out="WheelRolDir",
                                    )

def check_interpolation_back_and_forth(configFilePath):
    physical_domain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.0,15.0),
        "meshSize":1
    }

    physical_properties={
        "problem_type":"StaticMechanicalStandard",
        "materials":[["ALL", {"law":"LinearElasticity","young":5.98e6,"poisson":0.495} ]],
        "neumann":[["HOLE_BOUND", {"type": "RimRigidityNeumann", "Force": 1.0e7}]],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.0}] ]
    }
    simulator=GetfemSimulator(physical_domain=physical_domain,physical_properties=physical_properties)

    trainingInput={
              "Force":(1.0e4,1.0e7),
              }

    training_actor=LHSSampler(space_params=trainingInput)

    attr_names=(PFN.displacement,PFN.contactMultiplier)
    pneumatic_wheel_dataset_train=SamplerStaticWheelDataSet("train",attr_names=attr_names,attr_x= ("Force",),attr_y= ("disp",))
    path_out="WheelRegular"
    pneumatic_wheel_dataset_train.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=3,
                                    actor_seed=42,
                                    path_out=path_out
                                    )

    regular_dataset_reloaded=SamplerStaticWheelDataSet("train",attr_names=(PFN.displacement,PFN.contactMultiplier,"Force"),attr_x= ("Force",),attr_y= ("disp",))
    regular_dataset_reloaded.load(path=path_out)


    charac_sizes=[32,48,64,96,128]
    abs_error=[None]*len(charac_sizes)
    for charac_id,charac_size in enumerate(charac_sizes):
        print("Interpolation for charac_size=",charac_size)
        grid_support={"origin":(-16.0,0.0),"lenghts":(32.0,32.0),"sizes":(charac_size,charac_size)}
        interpolated_dataset_grid=DataSetInterpolatorOnGrid(name="train",simulator=simulator,
                                                    dataset=regular_dataset_reloaded,
                                                    grid_support=grid_support)
        dofnum_by_field={PFN.displacement:2}
        path_out="WheelInterpolated"
        interpolated_dataset_grid.generate(dofnum_by_field=dofnum_by_field,path_out=path_out)

        interpolated_dataset_grid_reloaded=DataSetInterpolatorOnGrid(name="train",simulator=simulator,
                                                    dataset=regular_dataset_reloaded,
                                                    grid_support=grid_support)
        interpolated_dataset_grid_reloaded.load(path=path_out)

        interpolated_dataset_mesh=DataSetInterpolatorOnMesh(name="train",simulator=simulator,
                                                    dataset=interpolated_dataset_grid_reloaded)
        path_out="WheelInterpolatedOnMesh"
        interpolated_dataset_mesh.generate(field_names=[PFN.displacement],path_out=path_out)

        interpolated_dataset_mesh_reloaded=DataSetInterpolatorOnMesh(name="train",simulator=simulator,
                                                    dataset=interpolated_dataset_grid_reloaded)
        interpolated_dataset_mesh_reloaded.load(path=path_out)

        original_input=np.squeeze(regular_dataset_reloaded.data["Force"])
        reinterpolated_input=interpolated_dataset_mesh_reloaded.accumulated_data_from_grid["Force"]
        np.testing.assert_allclose(original_input,reinterpolated_input)

        original_data=regular_dataset_reloaded.data["disp"]
        reinterpolated_data=interpolated_dataset_mesh_reloaded.interpolated_dataset["disp"]
        abs_error[charac_id]=np.linalg.norm(original_data-reinterpolated_data)

    #Check error is decreasing
    print(abs_error)
    np.testing.assert_equal(abs_error[::-1],np.sort(abs_error))

from lips import get_root_path

if __name__ == '__main__':
    configFilePath=get_root_path()+os.path.join("..","configurations","pneumatic","benchmarks","confWheel.ini")
    check_interpolation_back_and_forth(configFilePath=configFilePath)
    check_static_samples_generation(configFilePath=configFilePath)
    check_quasi_static_generation(configFilePath=configFilePath)