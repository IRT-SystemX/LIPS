"""
Usage:
    Introduce the sampling methods used to generate a space of parameters
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
import shutil
import numpy as np

def WriteDataset(dataset,path_out):
    """Save the internal data in a proper format

    Parameters
    ----------
    path_out: output path
        A str to indicate where to save the data.
    """
    full_path_out = os.path.join(os.path.abspath(path_out), dataset.name)

    if not os.path.exists(os.path.abspath(path_out)):
        os.mkdir(os.path.abspath(path_out))
        dataset.logger.info(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")

    if os.path.exists(full_path_out):
        dataset.logger.warning(f"Deleting previous run at {full_path_out}")
        shutil.rmtree(full_path_out)

    os.mkdir(full_path_out)
    dataset.logger.info(f"Creating the path {full_path_out} to store the dataset name {dataset.name}")

    for attr_nm in dataset._attr_names:
        np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data=dataset.data[attr_nm])

if __name__ == '__main__':
    import math

    from lips.physical_simulator.getfemSimulator import GetfemSimulator
    from lips.dataset.pneumaticWheelDataSetGenerators import PneumaticWheelDataSetStaticGenerator
    from lips.dataset.utils.sampler import LHSSampler
    import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

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
                                                            attr_outputs=(PFN.displacement,PFN.contactMultiplier),
                                                            attr_names=attr_names,
                                                            sampler=training_sampler,
                                                            nb_samples=5,
                                                            sampler_seed=42)

    pneumaticDataset=staticWheelGenerator.generate()
    WriteDataset(dataset=pneumaticDataset,path_out="FirstSaveDataset")