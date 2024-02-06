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
import numpy as np

from lips.dataset.pneumaticWheelDataSet import WheelDataSet

def ReadDataset(path_in,name,attr_names):
    """Load the internal data

    Parameters
    ----------
    path: input path
        A str to indicate where to load the data from.
    """
    if not os.path.exists(path_in):
        raise RuntimeError(f"{path} cannot be found on your computer")
    if not os.path.isdir(path_in):
        raise RuntimeError(f"{path} is not a valid directory")
    full_path = os.path.join(path_in, name)
    if not os.path.exists(full_path):
        raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                           f"a given `path_out` ?")

    for attr_nm in attr_names:
        path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
        if not os.path.exists(path_this_array):
            raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                               f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

    datasetFromData=WheelDataSet(name=name,
                                 attr_names=attr_names,
                                 attr_x= [],
                                 attr_y= [])

    if datasetFromData.data is not None:
        warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
    datasetFromData.data = {}
    datasetFromData.size = None

    for attr_nm in datasetFromData._attr_names:
        path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
        datasetFromData.data[attr_nm] = np.load(path_this_array)["data"]
        datasetFromData.size = datasetFromData.data[attr_nm].shape[0]

    inputs = {attr_x:datasetFromData.data[attr_x] for attr_x in datasetFromData._attr_x}
    datasetFromData._inputs = [dict(zip(inputs,t)) for t in zip(*inputs.values())]

    datasetFromData._infer_sizes()
    return datasetFromData

if __name__ == '__main__':
    import math
    from lips.physical_simulator.getfemSimulator import GetfemSimulator
    import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
    from lips.dataset.pneumaticWheelDataSetGenerators import PneumaticWheelDataSetQuasiStaticGenerator
    from lips.dataset.io.data_writer import WriteDataset

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

    quasiStatic_wheel_generator=PneumaticWheelDataSetQuasiStaticGenerator(name="Train",
                                                            simulator=training_simulator,
                                                            attr_inputs=("timeSteps",),
                                                            attr_outputs=("disp","contactMult"),
                                                            attr_names=attr_names)
    pneumatic_dataset=quasiStatic_wheel_generator.generate()
    print(pneumatic_dataset)
    WriteDataset(dataset=pneumatic_dataset,path_out="SecondSaveDataset")
    attr_names=[PFN.displacement,PFN.contactMultiplier,"timeSteps"]
    reloaded_dataset=ReadDataset(path_in=".",name="SecondSaveDataset/Train",attr_names=attr_names)
    print(reloaded_dataset)
    assert pneumatic_dataset==reloaded_dataset
