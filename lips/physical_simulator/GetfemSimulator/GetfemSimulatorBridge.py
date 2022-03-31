#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver
import lips.physical_simulator.GetfemSimulator.MeshGenerationTools as ExternalMesher
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemMecaProblem,GetfemRollingWheelProblem
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblemQuasiStatic import QuasiStaticRollingProblem,QuasiStaticMecanicalProblem

def MeshGeneration(physicalDomain):
    if physicalDomain["Mesher"]=="Getfem":
        return PhySolver.GenerateWheelMesh(wheelDimensions=physicalDomain["wheelDimensions"],\
                                    meshSize=physicalDomain["meshSize"],\
                                    RefNumByRegion=physicalDomain["refNumByRegion"])
    elif physicalDomain["Mesher"]=="Gmsh":
        return ExternalMesher.GenerateCoincidentHFLFMeshes(wheelExtMeshFile="wheel_ext",\
                                                           wheelMeshFile=physicalDomain["meshFilename"],\
                                                           interRadius=physicalDomain["interRadius"],\
                                                           wheelDim=physicalDomain["wheelDimensions"],\
                                                           meshSize=physicalDomain["meshSize"],\
                                                           version=physicalDomain["version"])
    else:
        raise Exception("Mesher "+str(physicalDomain["Mesher"])+" not supported")

def SimulatorGeneration(physicalDomain,physicalProperties):
    problemType=physicalProperties["ProblemType"]

    classNameByProblemType = {
                               "StaticMechanicalStandard":"GetfemMecaProblem",
                               "StaticMechanicalRolling":"GetfemRollingWheelProblem",
                               "QuasiStaticMechanicalStandard":"QuasiStaticRollingProblem",
                               "QuasiStaticMechanicalRolling":"QuasiStaticMecanicalProblem"
                               }

    try:
        simulator = globals()[classNameByProblemType[problemType]]()
    except KeyError:
        raise(Exception("Unable to treat this kind of problem !"))

    mesh=MeshGeneration(physicalDomain)
    simulator.mesh=mesh
    simulator.refNumByRegion=physicalDomain["refNumByRegion"]

    filterPhysicalProperties={k: v for k, v in physicalProperties.items() if k!="ProblemType"}
    for physicalProperty,physicalValue in filterPhysicalProperties.items():
        attribute=setattr(simulator,physicalProperty,physicalValue)

    return simulator


