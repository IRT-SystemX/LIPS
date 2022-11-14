#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This file introduce a Getfem++ interface.
#The idea is to combine the individual features relying on Getfem++ "bricks" to build a physical problem

import numpy as np
import math
import abc
from copy import deepcopy

import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemRollingWheelProblem,GetfemMecaProblem
import lips.physical_simulator.GetfemSimulator.GetfemBricks.ModelTools as gfModel
import lips.physical_simulator.GetfemSimulator.GetfemBricks.ExportTools as gfExport
from lips.physical_simulator.GetfemSimulator.GetfemBricks.RollingConditions import ComputeDISRollingCondition

class StaticSolverFailure(Exception):
    pass

class QuasiStaticSolverFailure(Exception):
    pass

class QuasiStaticMecaProblemBase(metaclass=abc.ABCMeta):
    def __init__(self,name=None,auxiliaryOutputs=None,other=None):
        if other is None:
            self.name=name
            self.auxiliaryOutputs=auxiliaryOutputs
            self._staticProblem=None
            self.transientParams=dict()
            self.solutions=dict()
            self.timeSteps=np.array([])
        else:
            self.name=other.name
            self.auxiliaryOutputs=other.auxiliaryOutputs
            self._staticProblem=type(other._staticProblem)(other=other._staticProblem)
            self.transientParams=deepcopy(other.transientParams)
            self.solutions=deepcopy(other.solutions)
            self.timeSteps=deepcopy(other.timeSteps)
        
    @property
    def mesh(self):
        return self._staticProblem.mesh

    @mesh.setter
    def mesh(self, problemMesh):
        self._staticProblem.mesh = problemMesh

    @property
    def refNumByRegion(self):
        return self._staticProblem.refNumByRegion

    @refNumByRegion.setter
    def refNumByRegion(self, problemRefNumByRegion):
        self._staticProblem.refNumByRegion = problemRefNumByRegion

    @property
    def materials(self):
        return self._staticProblem.materials

    @materials.setter
    def materials(self, problemMaterials):
        self._staticProblem.materials = problemMaterials

    @property
    def dirichlet(self):
        return self._staticProblem.dirichlet

    @dirichlet.setter
    def dirichlet(self, problemDirichlet):
        self._staticProblem.dirichlet = problemDirichlet

    @property
    def neumann(self):
        return self._staticProblem.neumann

    @neumann.setter
    def neumann(self, problemNeumann):
        self._staticProblem.neumann = problemNeumann

    @property
    def contact(self):
        return self._staticProblem.contact

    @contact.setter
    def contact(self, problemContact):
        self._staticProblem.contact = problemContact

    @property
    def incompressibility(self):
        return self._staticProblem.incompressibility

    @incompressibility.setter
    def incompressibility(self, problemIncompressibility):
        self._staticProblem.incompressibility = problemIncompressibility

    def Preprocessing(self):
        self._staticProblem.Preprocessing()

    def BuildModel(self):
        self._staticProblem.BuildModel()

    def InitSolution(self):
        self.solutions={fieldType: [] for _,fieldType,_ in self._staticProblem.spacesVariables}

    def InitVariable(self,fieldType,field):
        self._staticProblem.InitVariable(fieldType,field)

    def GetSolution(self,fieldType):
        return self.solutions[fieldType]

    def GetBasicCoordinates(self):
        return self._staticProblem.GetBasicCoordinates()

    def GetFeSpace(self,fieldType):
        return self._staticProblem.GetFeSpace(fieldType)

    def RunProblem(self,options=None):
        self.InitSolution()
        timeParams={"step":0.0,"currentTime":0.0}
        self.RunQuasiStaticClassicalSolve(timeParams=timeParams,options=options)

    def RunQuasiStaticClassicalSolve(self,timeParams,options=None):
        finalTime,dt=self.transientParams["time"],self.transientParams["timeStep"]
        step,currentTime=timeParams["step"],timeParams["currentTime"]
        timeSteps = []
        while currentTime < finalTime:
            currentTime=round(step*dt,10)
            print("Current time: ",currentTime)
            self.UpdateStaticProblem(currentTime)
            try:
                self.SolveStaticProblem(options=options)
            except StaticSolverFailure:
                raise QuasiStaticSolverFailure("Static resolution has failed at step {} and at time {}".format(step,currentTime))
            self.SaveStaticSolution()
            timeSteps.append(currentTime)
            step+=1

        self.timeSteps=np.array(timeSteps)

    @abc.abstractmethod
    def UpdateStaticProblem(self,time):
        pass

    def AtomicSolve(self,options=None):
        self.SolveStaticProblem(options)
        u = self._staticProblem.GetSolution(PFN.displacement)
        return u

    def SolveStaticProblem(self,options=None):
        state=self._staticProblem.RunProblem(options=options)
        if not state:
            raise StaticSolverFailure("Static solve has failed")

    def SaveStaticSolution(self):
        for variable,fieldType,_ in self._staticProblem.spacesVariables:
            staticSolution=gfModel.GetModelVariableValue(self._staticProblem.model,variable)
            self.solutions[fieldType].append(staticSolution)

    def __getattr__(self, name):
        def attrchecker(name):
            return lambda obj: hasattr(obj, name)

        possible = next(filter(attrchecker(name), [self._staticProblem]), None)
        if possible:
            return possible.__getattribute__(name)
        raise AttributeError("No attribute {!s} found".format(name))

    def ExportSolutionInCsv(self,filename,outputformat):
        feSpace,solutions=self.GetFeSpace(PFN.displacement),self.GetSolution(PFN.displacement)
        gfExport.ExportTransientSolutionInCSV(filename,outputformat,feSpace,solutions)

    def ExportTimeStepsInCsv(self,filename):
        np.savetxt(filename, self.timeSteps, header="time", delimiter=',',comments='')

    def ExportSolution(self,filename,extension):
        if extension=="gmsh":
            self.ExportSolutionInGmsh(filename=filename)
        elif extension=="vtk":
            self.ExportSolutionInVTKSplitFiles(filename=filename)
        else:
            raise Exception("Extension "+str(extension)+" not recognized")

    def ExportSolutionInGmsh(self,filename,fieldType=PFN.displacement):
        feSpace,solutions=self.GetFeSpace(fieldType),self.GetSolution(fieldType)
        gfExport.ExportTransientPrimalSolutionInGmsh(filename,feSpace,solutions)

    def ExportSolutionInVTKSplitFiles(self,filename,fieldType=PFN.displacement):
        feSpace,solutions=self.GetFeSpace(fieldType),self.GetSolution(fieldType)
        gfExport.ExportTransientPrimalSolutionInVTKSplitFiles(filename,feSpace,solutions)

    def ExportSolutionWithMultipliers(self,filename,extension):
        if not self._staticProblem.contact:
            raise Exception("There is no contact condition in the static problem!")
        else: 
            contactBoundaryRegion=self._staticProblem.contact[0][0]
            if len(self._staticProblem.contact)>1:
                print("Warning: Only export multipliers associated to the contact boundary "+str(contactBoundaryRegion))

        if extension=="gmsh":
            self.ExportSolutionWithMultipliersInGmsh(filename=filename,contactRegion=contactBoundaryRegion)
        elif extension=="vtk":
            self.ExportSolutionWithMultipliersInVtk(filename=filename,contactRegion=contactBoundaryRegion)
        else:
            raise Exception("Extension "+str(extension)+" not recognized")

    def ExportSolutionWithMultipliersInVtk(self,filename,contactRegion):
        feSpaces = self.GetFeSpace(PFN.displacement),self.GetFeSpace(PFN.contactMultiplier)
        solutions = self.GetSolution(PFN.displacement),self.GetSolution(PFN.contactMultiplier)
        gfExport.ExportTransientPrimalDualSolutionInVtk(filename,feSpaces,solutions,self._staticProblem.refNumByRegion[contactRegion])

    def ExportSolutionWithMultipliersInGmsh(self,filename,contactRegion):
        feSpaces = self.GetFeSpace(PFN.displacement),self.GetFeSpace(PFN.contactMultiplier)
        solutions = self.GetSolution(PFN.displacement),self.GetSolution(PFN.contactMultiplier)
        gfExport.ExportTransientPrimalDualSolutionInGmsh(filename,feSpaces,solutions,self._staticProblem.refNumByRegion[contactRegion])


class QuasiStaticRollingProblem(QuasiStaticMecaProblemBase):
    def __init__(self,name=None,auxiliaryOutputs=None,other=None):
        super(QuasiStaticRollingProblem,self).__init__(name,auxiliaryOutputs,other)
        if other is None:
            self._staticProblem=GetfemRollingWheelProblem(name=name,auxiliaryOutputs=auxiliaryOutputs)

    @property
    def rolling(self):
        return self._staticProblem.rolling

    @rolling.setter
    def rolling(self, problemRolling):
        self._staticProblem.rolling = problemRolling

    def BuildModel(self):
        self._staticProblem.rolling[1]['currentTime'] = 0.0
        self._staticProblem.BuildModel()

    def ExportSolutionAtInstantkInVTK(self, filename, fieldType=PFN.displacement):
        feSpace, solutions = self.GetFeSpace(PFN.displacement), self.GetSolution(fieldType)
        gfExport.ExportTransientPrimalSolutionInVTK(filename, feSpace, [solutions[-1]])

    def UpdateStaticProblem(self,time):
        self.UpdateRollingCondition(time)

    def UpdateRollingCondition(self,time):
        tagname,roll=self._staticProblem.rolling
        rollingParams={k: roll[k] for k in set(list(roll.keys())) - set(["type"])}
        problemRollingConditionParams={"model":self._staticProblem.model,
                                       "feSpace":self.GetFeSpace(PFN.displacement),
                                       "dimension":self.dim}
        dispRollingConditionParams={"theta_Rolling":rollingParams["theta_Rolling"],
                                    "d":rollingParams["d"],
                                    "currentTime":time}

        newRolling=ComputeDISRollingCondition(self._staticProblem.refNumByRegion[tagname],
                                              problemRollingConditionParams,
                                              dispRollingConditionParams)
        gfModel.SetModelVariableValue(model=self._staticProblem.model, variableName='rollingRHS', variableValue=newRolling)

    def __str__(self):
        s="Quasi-static rolling problem\n"
        for paramName,paramVal in self.transientParams.items():
            s+="\t"+str(paramName)+": "+str(paramVal)+"\n"
        s+=self._staticProblem.__str__()+"\n"
        return s

class QuasiStaticForceRollingProblem(QuasiStaticRollingProblem):
    def __init__(self,name=None,auxiliaryOutputs=None,other=None):
        super(QuasiStaticForceRollingProblem,self).__init__(name,auxiliaryOutputs,other)

    def RunProblem(self):
        self.InitSolution()
        disp_w_c=self.ComputeDisplacementAfterLoadingStep()
        self.UpdateInitialRolllingCondition(initDisp=disp_w_c)
        firsInstant = self.transientParams["timeStep"]
        timeParams={"step":1, "currentTime":firsInstant}
        self.RunQuasiStaticClassicalSolve(timeParams)

    def ComputeDisplacementAfterLoadingStep(self):
        self._staticProblem.model.disable_bricks(self._staticProblem.rollingBrick)
        self.SolveStaticProblem()
        self.SaveStaticSolution()

        disp_w_c = -self._staticProblem.model.variable('alpha_D')[0]
        return disp_w_c

    def UpdateInitialRolllingCondition(self,initDisp):
        self._staticProblem.model.disable_bricks(self._staticProblem.loadingBrick)
        self._staticProblem.model.enable_bricks(self._staticProblem.rollingBrick)
        self._staticProblem.model.disable_variable('alpha_D')
        self.rolling[1]['d'] = initDisp

class QuasiStaticMecanicalProblem(QuasiStaticMecaProblemBase):
    def __init__(self,name=None,auxiliaryOutputs=None,other=None):
        super(QuasiStaticMecanicalProblem,self).__init__(name=name,auxiliaryOutputs=auxiliaryOutputs,other=other)
        if other is None:
            self.enforcedCondition=""
            self._staticProblem=GetfemMecaProblem()
        else:
            self.enforcedCondition=other.enforcedCondition

    def UpdateStaticProblem(self,time):
        if self.enforcedCondition =="Dirichlet":
            self.UpdateDirichletConditions(time)
        elif self.enforcedCondition =="Neumann":
            self.UpdateNeumannConditions(time)
        else:
            raise Exception("Should not pass here!")

    def UpdateDirichletConditions(self,time):
        for dirichId,dirich in enumerate(self._staticProblem.dirichlet):
            if dirich[1]["type"]=="scalar":
                newAmplitude,angleDisp=time*dirich[1]["Disp_Amplitude"],dirich[1]["Disp_Angle"]
                newDirichlet=[newAmplitude*math.cos(angleDisp),newAmplitude*math.sin(angleDisp)]
                gfModel.SetModelVariableValue(model=self._staticProblem.model,variableName='DirichletData'+str(dirichId),variableValue=newDirichlet)
            else:
                raise Exception("Not handled yet")

    def UpdateNeumannConditions(self,time):
        for neumId,neum in enumerate(self._staticProblem.neumann):
            if neum[1]["type"]=="StandardNeumann":
                newFx,newFy=time*neum[1]["fx"],time*neum[1]["fy"]
                gfModel.SetModelVariableValue(model=self._staticProblem.model,variableName='NeumannData'+str(neumId),variableValue=[newFx,newFy])
            else:
                raise Exception("Not handled yet")

    def __str__(self):
        s="Quasi-static problem\n"
        s+="\t"+self.enforcedCondition+" piloted\n"
        for paramName,paramVal in self.transientParams.items():
            s+="\t"+str(paramName)+": "+str(paramVal)+"\n"
        s+=self._staticProblem.__str__()+"\n"
        return s

#################################Test#################################
import lips.physical_simulator.GetfemSimulator.GetfemBricks.MeshTools as gfMesh
from lips.physical_simulator.GetfemSimulator.MeshGenerationTools import Standard3DWheelGenerator

def CheckIntegrity_BeamProblemQuasiStaticNeumann():
    WPMQuasi=QuasiStaticMecanicalProblem()
    refNumByRegion={"Left":1,"Bottom":2,"Right":3,"Top":4}
    WPMQuasi.mesh=gfMesh.Generate2DBeamMesh(meshSize=5,RefNumByRegion=refNumByRegion)
    WPMQuasi.refNumByRegion=refNumByRegion
    WPMQuasi.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPMQuasi.dirichlet=[["Left",{"type" : "scalar", "Disp_Amplitude":0, "Disp_Angle":0}] ]
    WPMQuasi.neumann=[["Top",{"type" : "StandardNeumann", "fx":0.0,"fy":-200}] ]
    WPMQuasi.transientParams={"time":1.0,"timeStep":0.5}
    WPMQuasi.enforcedCondition="Neumann"

    WPMQuasi.Preprocessing()
    WPMQuasi.BuildModel()
    print(WPMQuasi)
    WPMQuasi.RunProblem()
    WPMQuasi.ExportSolution(filename="QuasiStaticBeamNeumann.pos",extension="gmsh")
    WPMQuasi.ExportSolution(filename="QuasiStaticBeamNeumann.vtk",extension="vtk")
    WPMQuasi.ExportTimeStepsInCsv(filename="BeamNeumannTimeSteps.csv")
    solution=WPMQuasi.GetSolution(PFN.displacement)
    solutionIsPhysical= np.max(solution) < 1
    assert solutionIsPhysical

    instanceAdress=WPMQuasi.__repr__()

    WPMQuasi2=type(WPMQuasi)(other=WPMQuasi)
    instanceAdress2=WPMQuasi2.__repr__()
    assert instanceAdress != instanceAdress2

    WPMQuasi2.solution=None
    WPMQuasi2.BuildModel()
    WPMQuasi2.RunProblem()
    solution2=WPMQuasi2.GetSolution(PFN.displacement)
    np.testing.assert_array_almost_equal(solution,solution2)
    return "OK"


def CheckIntegrity_BeamProblemQuasiStaticDirichlet():
    WPMQuasi=QuasiStaticMecanicalProblem()
    refNumByRegion={"Left":1,"Bottom":2,"Right":3,"Top":4}
    WPMQuasi.mesh=gfMesh.Generate2DBeamMesh(meshSize=5,RefNumByRegion=refNumByRegion)
    WPMQuasi.refNumByRegion=refNumByRegion
    WPMQuasi.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPMQuasi.dirichlet=[["Left",{"type" : "scalar", "Disp_Amplitude":0, "Disp_Angle":0}],["Right",{"type" : "scalar", "Disp_Amplitude":1, "Disp_Angle":-math.pi/2}] ]
    WPMQuasi.transientParams={"time":1.0,"timeStep":0.5}
    WPMQuasi.enforcedCondition="Dirichlet"
    WPMQuasi.Preprocessing()
    WPMQuasi.BuildModel()
    print(WPMQuasi)
    WPMQuasi.RunProblem()
    WPMQuasi.ExportSolution(filename="QuasiStaticBeamDirichlet.pos",extension="gmsh")
    return "OK"

def CheckIntegrity_2DDispRollingWheelProblem():
    WPMQuasi=QuasiStaticRollingProblem()
    wheelDimensions=(8., 15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPMQuasi.mesh=gfMesh.GenerateWheelMeshRolling(wheelDimensions=wheelDimensions,meshSize=2,RefNumByRegion=refNumByRegion)
    WPMQuasi.refNumByRegion=refNumByRegion
    WPMQuasi.materials = [["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1}]]
    WPMQuasi.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPMQuasi.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.6}] ]
    WPMQuasi.rolling=["HOLE_BOUND",{"type" : "DIS_Rolling", "theta_Rolling":150., 'd': 1.}]
    dt = 10e-4
    WPMQuasi.transientParams={"time": 5*dt, "timeStep": dt}
    
    WPMQuasi.Preprocessing()
    WPMQuasi.BuildModel()
    print(WPMQuasi)

    options={"max_iter":2,
            "max_res":1e-8}
    np.testing.assert_raises(QuasiStaticSolverFailure, WPMQuasi.RunProblem, options)

    WPMQuasi.RunProblem()
    filenameSuffix="2DDispRollingWheel"
    WPMQuasi.ExportSolution(filename=filenameSuffix+".pos",extension="gmsh")
    WPMQuasi.ExportSolution(filename=filenameSuffix+".vtk",extension="vtk")
    WPMQuasi.ExportSolutionWithMultipliers(filename=filenameSuffix+"Pretty.pos",extension="gmsh")
    WPMQuasi.ExportSolutionWithMultipliers(filename=filenameSuffix+"Pretty",extension="vtk")
    WPMQuasi.ExportSolutionInCsv(filename=filenameSuffix+"Displacement",outputformat="VectorSolution")
    WPMQuasi.ExportTimeStepsInCsv(filename=filenameSuffix+"TimeSteps.csv")

    solution=WPMQuasi.GetSolution(PFN.displacement)
    instanceAdress=WPMQuasi.__repr__()

    WPMQuasi2=type(WPMQuasi)(other=WPMQuasi)
    instanceAdress2=WPMQuasi2.__repr__()
    assert instanceAdress != instanceAdress2

    WPMQuasi2.solution=None
    WPMQuasi2.BuildModel()
    WPMQuasi2.RunProblem()
    solution2=WPMQuasi2.GetSolution(PFN.displacement)
    np.testing.assert_array_almost_equal(solution,solution2)
    return "OK"

def CheckIntegrity_2DTwoStepForcRollingWheelProblem():
    WPMQuasi = QuasiStaticForceRollingProblem()
    wheelDimensions = (8., 15.)
    refNumByRegion = {"HOLE_BOUND": 1, "CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPMQuasi.mesh = gfMesh.GenerateWheelMeshRolling(wheelDimensions=wheelDimensions, meshSize=2,
                                                       RefNumByRegion=refNumByRegion)
    WPMQuasi.refNumByRegion = refNumByRegion
    WPMQuasi.materials = [["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1}]]
    WPMQuasi.contact = [["CONTACT_BOUND", {"type": "Plane", "gap": 0.0, "fricCoeff": 0.6}]]
    WPMQuasi.rolling = ["HOLE_BOUND", {"type": "TwoSteps_FORC_Rolling", "theta_Rolling": 150., "Force": 1.99E1,
                                       'penalization': 1, 'd': 0,"radius":8}]
    dt = 10e-4
    WPMQuasi.transientParams = {"time": 5 * dt, "timeStep": dt}
    WPMQuasi.Preprocessing()
    WPMQuasi.BuildModel()
    WPMQuasi.RunProblem()
    WPMQuasi.ExportSolution(filename="2DForcedRollingWheel.vtk", extension="vtk")
    WPMQuasi.ExportSolution(filename="2DForcedRollingWheel.pos",extension="gmsh")

    return "OK"

def CheckIntegrity_3DDispRollingWheelProblem():
    mesh3DConfig={"wheel_Dimensions":(8.,15.,4),
                "mesh_size":2.0}
    gmshMesh=Standard3DWheelGenerator(**mesh3DConfig)
    gmshMesh.GenerateMesh(outputFile="my3DWheelForDisp")

    mesh=gfMesh.ImportGmshMesh(meshFile="my3DWheelForDisp.msh")
    WPMQuasi=QuasiStaticRollingProblem()
    WPMQuasi.mesh=mesh
    WPMQuasi.refNumByRegion=gmshMesh.tagMap
    WPMQuasi.materials = [["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1}]]
    WPMQuasi.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0,"source_z":0.0}] ]
    WPMQuasi.contact=[ ["Exterior",{"type" : "Plane","gap":0.0,"fricCoeff":0.6}] ]
    WPMQuasi.rolling=["Interior",{"type" : "DIS_Rolling", "theta_Rolling":150., 'd': 1.}]
    dt = 10e-4
    WPMQuasi.transientParams={"time": 5*dt, "timeStep": dt}
    
    WPMQuasi.Preprocessing()
    WPMQuasi.BuildModel()
    WPMQuasi.RunProblem()
    WPMQuasi.ExportSolution(filename="3DDispRollingWheel.vtk", extension="vtk")

    return "OK"

def CheckIntegrity_3DTwoStepForcRollingWheelProblem():
    mesh3DConfig={"wheel_Dimensions":(8.,15.,4),
                "mesh_size":2.0}
    gmshMesh=Standard3DWheelGenerator(**mesh3DConfig)
    gmshMesh.GenerateMesh(outputFile="my3DWheelForForc")

    mesh=gfMesh.ImportGmshMesh(meshFile="my3DWheelForForc.msh")

    WPMQuasi = QuasiStaticForceRollingProblem()
    WPMQuasi.mesh = mesh
    WPMQuasi.refNumByRegion = gmshMesh.tagMap
    WPMQuasi.materials = [["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1}]]
    WPMQuasi.contact = [["Exterior", {"type": "Plane", "gap": 0.0, "fricCoeff": 0.6}]]
    WPMQuasi.rolling = ["Interior", {"type": "TwoSteps_FORC_Rolling", "theta_Rolling": 150., "Force": 1.99E1,
                                       'penalization': 1, 'd': 0,"radius":8}]
    dt = 10e-4
    WPMQuasi.transientParams = {"time": 5 * dt, "timeStep": dt}
    WPMQuasi.Preprocessing()
    WPMQuasi.BuildModel()
    WPMQuasi.RunProblem()
    WPMQuasi.ExportSolution(filename="3DForcedRollingWheel.vtk", extension="vtk")
    WPMQuasi.ExportSolution(filename="3DForcedRollingWheel.pos",extension="gmsh")
    return "OK"


def CheckIntegrity():
    totest = [
    CheckIntegrity_BeamProblemQuasiStaticNeumann,
    CheckIntegrity_BeamProblemQuasiStaticDirichlet,
    CheckIntegrity_2DDispRollingWheelProblem,
    CheckIntegrity_2DTwoStepForcRollingWheelProblem,
    CheckIntegrity_3DDispRollingWheelProblem,
    CheckIntegrity_3DTwoStepForcRollingWheelProblem
              ]

    for test in totest:
        res =  test()
        if  res.lower() != "ok" :
            return res

    return "OK"

if __name__ =="__main__":
    CheckIntegrity()
