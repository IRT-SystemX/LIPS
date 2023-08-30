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

import numpy as np
import copy
from typing import Union
from collections.abc import Iterable
from tqdm import tqdm 

from lips.logger import CustomLogger
from lips.config.configmanager import ConfigManager
from lips.dataset.datasetGeneratorBase import DataSetGeneratorBase
from lips.dataset.pneumaticWheelDataSet import WheelDataSet
from lips.dataset.utils.sampler import LHSSampler
from lips.physical_simulator.getfemSimulator import GetfemSimulator

class PneumaticDataSetGeneratorBase(DataSetGeneratorBase):
    def __init__(self,
                 name:str,
                 simulator:GetfemSimulator,
                 attr_inputs:Iterable,
                 attr_outputs:Iterable,
                 attr_names:Iterable,
                 nb_samples:int,
                 log_path: Union[str, None]=None):
        super(PneumaticDataSetGeneratorBase,self).__init__(name=name,
                                                           simulator=simulator,
                                                           attr_inputs=attr_inputs,
                                                           attr_outputs=attr_outputs,
                                                           attr_names=attr_names,
                                                           nb_samples=nb_samples,
                                                           log_path=log_path)

        self._dataset_type=WheelDataSet
        try:
            import getfem
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` a wheel dateset  if you don't have "
                               "the getfem package installed") from exc_
        if nb_samples <= 0:
            raise RuntimeError("Impossible to generate a negative number of data.")

    def _init_data(self, simulator:GetfemSimulator, nb_samples:int):
        simulator.build_model()
        for attr_nm in self._attr_names:
            variableInitVal = simulator.get_variable_value(field_name=attr_nm)
            self._data[attr_nm] = np.zeros((nb_samples, variableInitVal.shape[0]), dtype=variableInitVal.dtype)

    def _load_dataset_from_store_data(self):
        datasetFromData=self._dataset_type(name=self._name,
                                     attr_names=self._attr_names,
                                     attr_x= self._attr_inputs,
                                     attr_y= self._attr_outputs)

        attr_names_to_keep=self._attr_inputs+self._attr_outputs
        datasetFromData._attr_names=self._attr_inputs+self._attr_outputs
        datasetFromData.load_from_data(data=self._data,attr_names_to_keep=attr_names_to_keep)
        return datasetFromData

class PneumaticWheelDataSetStaticGenerator(PneumaticDataSetGeneratorBase):
    def __init__(self,
                 name:str,
                 simulator:GetfemSimulator,
                 attr_inputs:Iterable,
                 attr_outputs:Iterable,
                 attr_names:Iterable,
                 sampler:LHSSampler,
                 nb_samples:int,
                 sampler_seed: Union[None, int]=None,
                 log_path: Union[str, None]=None):
        super(PneumaticWheelDataSetStaticGenerator,self).__init__(name=name,
                                                                  simulator=simulator,
                                                                  attr_inputs=attr_inputs,
                                                                  attr_outputs=attr_outputs,
                                                                  attr_names=attr_names,
                                                                  nb_samples=nb_samples,
                                                                  log_path=log_path)

        self._sampler=sampler
        self._sampler_seed=sampler_seed

    def generate(self):
        self._generate_inputs()
        self._init_data(simulator=self._simulator, nb_samples=self._nb_samples)
        self._generate_data()
        dataset=self._load_dataset_from_store_data()
        return dataset

    def _generate_inputs(self):
        self._inputs=self._sampler.generate_samples(nb_samples=self._nb_samples,sampler_seed=self._sampler_seed)

    def _generate_data(self):
        for current_size,sample in enumerate(tqdm(self._inputs, desc=self._name)):
            simulator=type(self._simulator)(simulator_instance=self._simulator)
            simulator.modify_state(state=sample)
            simulator.build_model()
            solverState=simulator.run_problem()
            self._store_obs(current_size=current_size,obs=simulator)

        sampler_attrib=self._sampler.get_attributes_as_data()
        self._data={**self._data, **sampler_attrib}


    def _store_obs(self, current_size: int, obs:GetfemSimulator):
        for attr_nm in self._attr_names:
            solution = obs.get_solution(field_name=attr_nm)
            self._data[attr_nm][current_size, :] = solution


class PneumaticWheelDataSetQuasiStaticGenerator(PneumaticDataSetGeneratorBase):
    def __init__(self,
                 name:str,
                 simulator:GetfemSimulator,
                 attr_inputs:Iterable,
                 attr_outputs:Iterable,
                 attr_names:Iterable,
                 log_path: Union[str, None]=None):
        transientParams=getattr(simulator._simulator,"transientParams")
        nb_samples=int(transientParams["time"]//transientParams["timeStep"]) + 1
        super(PneumaticWheelDataSetQuasiStaticGenerator,self).__init__(name=name,
                                                                  simulator=simulator,
                                                                  attr_inputs=attr_inputs,
                                                                  attr_outputs=attr_outputs,
                                                                  attr_names=attr_names,
                                                                  nb_samples=nb_samples,
                                                                  log_path=log_path)


    def generate(self):
        self._init_data(simulator=self._simulator, nb_samples=self._nb_samples)
        self._generate_data()
        dataset=self._load_dataset_from_store_data()
        return dataset

    def _generate_data(self):
        solverState=self._simulator.run_problem()
        self._store_obs(obs=self._simulator)

    def _store_obs(self, obs:GetfemSimulator):
        timesteps=getattr(obs._simulator,"timeSteps")
        self._data["timeSteps"]=timesteps
        for attr_nm in self._attr_names:
            array_ = obs.get_solution(field_name=attr_nm)
            self._data[attr_nm] = np.array(array_)


import math
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

def check_static_samples_generation():
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

    training_sampler=LHSSampler(space_params=trainingInput)
    attr_names=(PFN.displacement,PFN.contactMultiplier)

    staticWheelGenerator=PneumaticWheelDataSetStaticGenerator(name="Train",
                                                            simulator=training_simulator,
                                                            attr_inputs=("young","poisson","fricCoeff"),
                                                            attr_outputs=("disp",),
                                                            attr_names=attr_names,
                                                            sampler=training_sampler,
                                                            nb_samples=5,
                                                            sampler_seed=42)

    pneumaticDataset=staticWheelGenerator.generate()
    print(pneumaticDataset)
    extract_x, extract_y=pneumaticDataset.extract_data()
    print(extract_x)
    print(extract_y)


def check_quasi_static_generation():
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

    quasiStaticWheelGenerator=PneumaticWheelDataSetQuasiStaticGenerator(name="Train",
                                                            simulator=training_simulator,
                                                            attr_inputs=("timeSteps",),
                                                            attr_outputs=("disp","contactMult"),
                                                            attr_names=attr_names)
    pneumaticDataset=quasiStaticWheelGenerator.generate()
    print(pneumaticDataset)

if __name__ == '__main__':
    check_static_samples_generation()
    check_quasi_static_generation()
