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

from typing import Union
import shutil

import numpy as np

from lips.dataset.pneumaticWheelDataSet import WheelDataSet
from lips.physical_simulator.getfemSimulator import GetfemSimulator
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
from lips.physical_simulator.GetfemSimulator.GetfemSimulatorBridge import GetfemInterpolationOnSupport,InterpolationOnCloudPoints


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
                 dataset:WheelDataSet,
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
                 dataset:WheelDataSet):
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

from lips.dataset.pneumaticWheelDataSetGenerators import PneumaticWheelDataSetStaticGenerator
from lips.dataset.utils.sampler import LHSSampler

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

    training_sampler=LHSSampler(space_params=trainingInput)
    attr_names=(PFN.displacement,PFN.contactMultiplier)
    staticWheelGenerator=PneumaticWheelDataSetStaticGenerator(name="train",
                                                                  simulator=simulator,
                                                                  attr_inputs=("Force",),
                                                                  attr_outputs=("disp",),
                                                                  attr_names=(PFN.displacement,),
                                                                  sampler=training_sampler,
                                                                  nb_samples=3,
                                                                  sampler_seed=42)
    regular_dataset_reloaded=staticWheelGenerator.generate()


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

from lips import GetRootPath
import os

if __name__ == '__main__':
    configFilePath=GetRootPath()+os.path.join("..","configurations","pneumatic","benchmarks","confWheel.ini")
    check_interpolation_back_and_forth(configFilePath=configFilePath)