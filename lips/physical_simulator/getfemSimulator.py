#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lips.physical_simulator.physicalSimulator import PhysicalSimulator
from lips.physical_simulator.GetfemSimulator.GetfemSimulatorBridge import SimulatorGeneration

def lipsToGetfemBridge(physicalDomain,physicalProperties):
    simulator=SimulatorGeneration(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    return simulator

class GetfemSimulator(PhysicalSimulator):
    """
    This simulator uses the `Getfem` library to implement a physical simulator.
    """
    def __init__(self, physicalDomain,physicalProblem):
        self._simulator = lipsToGetfemBridge(physicalDomain,physicalProperties)
        
        self._simulator.Preprocessing()

    def build_model(self):
        self._simulator.BuildModel()

    def run_problem(self):
        self._simulator.RunProblem()

    def get_state(self):
        """
        TODO
        """
        return self._simulator.internalInitParams

    def modify_state(self, actor):
        """
        TODO
        """
        self._simulator.SetPhyParams(actor)

if __name__ == '__main__':
    physicalDomain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.,15.),
        "meshSize":1
    }

    physicalProperties={
        "ProblemType":"StaticMechanicalStandard",
        "materials":[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]],
        "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":-5e3}] ],
        "dirichlet":[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":0, "Disp_Angle":0}] ],
    }
    mySimulator = GetfemSimulator(physicalDomain=physicalDomain,physicalProblem=physicalProperties)
    mySimulator.build_model()
    mySimulator.run_problem()