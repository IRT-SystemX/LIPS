#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import shutil

import copy
from typing import Union
from tqdm import tqdm  # TODO remove for final push

from lips.dataset.dataSet import DataSet
from lips.logger import CustomLogger

class RollingWheelDataSet(DataSet):
    """
    This specific DataSet uses Getfem framework to simulate data arising from a rolling wheel problem.
    """

    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 log_path: Union[str, None]=None
                 ):
        DataSet.__init__(self, name=name)
        self._attr_names = copy.deepcopy(attr_names)
        self.size = 0

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger

    def generate(self,
                 simulator: "GetfemSimulator",
                 actor,
                 path_out,
                 nb_samples,
                 simulator_seed: Union[None, int] = None,
                 actor_seed: Union[None, int] = None):
        """
        For this dataset, we use a GetfemSimulator and a Sampler to generate data from a rolling wheel.

        Parameters
        ----------
        simulator:
           In this case, this should be a grid2op environment

        actor:
           In this case, it is the sampler used for the input parameters space discretization

        path_out:
            The path where the data will be saved

        nb_samples:
            Number of rows (examples) in the final dataset

        simulator_seed:
            Seed used to set the simulator for reproducible experiments

        actor_seed:
            Seed used to set the actor for reproducible experiments

        Returns
        -------

        """
        try:
            import getfem
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` rolling wheel datet if you don't have "
                               "the getfem package installed") from exc_
        if nb_samples <= 0:
            raise RuntimeError("Impossible to generate a negative number of data.")

        samples=actor.generate_samples(nb_samples=nb_samples)
        self._init_store_data(simulator=simulator,nb_samples=nb_samples)

        for current_size,sample in enumerate(tqdm(samples, desc=self.name)):
            simulator=type(simulator)(simulatorInstance=simulator)
            simulator.modify_state(actor=sample)
            simulator.build_model()
            solverState=simulator.run_problem()
            
            self._store_obs(current_size=current_size,obs=simulator)

        self.size = nb_samples

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)

    def _init_store_data(self,simulator,nb_samples):
        self.data=dict()
        for attr_nm in self._attr_names:
            array_ = simulator.get_variable_value(field_name=attr_nm)
            truc= np.zeros((nb_samples, array_.shape[0]))
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)

    def _store_obs(self, current_size, obs):
        for attr_nm in self._attr_names:
            array_ = obs.get_solution(field_name=attr_nm)
            self.data[attr_nm][current_size, :] = array_

    def _save_internal_data(self, path_out):
        """save the self.data in a proper format"""
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

    def load(self, path):
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
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.data[attr_nm] = np.load(path_this_array)["data"]
            self.size = self.data[attr_nm].shape[0]

    def get_data(self, index):
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
        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)

        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el][:] = self.data[el][index, :]

        return res

if __name__ == '__main__':
    import math
    from lips.physical_simulator.getfemSimulator import GetfemSimulator
    physicalDomain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.,15.),
        "meshSize":1
    }

    physicalProperties={
        "ProblemType":"StaticMechanicalStandard",
        "materials":[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]],
        "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0}] ],
        "dirichlet":[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    }
    training_simulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)

    from lips.dataset.sampler import LHSSampler
    trainingInput={
              "young":(75.0,85.0),
              "poisson":(0.38,0.44),
              "fricCoeff":(0.5,0.8)
              }

    training_actor=LHSSampler(space_params=trainingInput)
    nb_sample_train=10
    path_datasets="TotoDir"

    import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
    attr_names=(PFN.displacement,PFN.contactMultiplier)

    rollingWheelDataSet=RollingWheelDataSet("train",attr_names=attr_names)
    rollingWheelDataSet.generate(simulator=training_simulator,
                                    actor=training_actor,
                                    path_out=path_datasets,
                                    nb_samples=nb_sample_train,
                                    actor_seed=42
                                    )
    print(rollingWheelDataSet.get_data(index=0))
    print(rollingWheelDataSet.data)