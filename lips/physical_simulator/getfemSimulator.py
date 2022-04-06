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
    def __init__(self, physicalDomain=None,physicalProperties=None,simulatorInstance=None):
        if simulatorInstance is None:
            self._simulator = lipsToGetfemBridge(physicalDomain,physicalProperties)
            self._simulator.Preprocessing()
        else:
            self._simulator=type(simulatorInstance._simulator)(simulatorInstance._simulator)

    def build_model(self):
        self._simulator.BuildModel()

    def run_problem(self):
        self._simulator.RunProblem()

    def get_solution(self,field_name):
        return self._simulator.GetSolution(field_name)

    def get_variable_value(self,field_name):
        return self._simulator.GetVariableValue(field_name)

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
    import math
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

    mySimulator = GetfemSimulator(physicalDomain=physicalDomain,physicalProblem=physicalProperties)
    mySimulator.build_model()
    mySimulator.run_problem()