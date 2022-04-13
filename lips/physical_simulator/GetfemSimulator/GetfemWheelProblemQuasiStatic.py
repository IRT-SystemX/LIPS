#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This file introduce a Getfem++ interface.
#The idea is to combine the individual features relying on Getfem++ "bricks" to build a physical problem

import numpy as np
import math
import abc
from copy import copy,deepcopy

import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemRollingWheelProblem,GetfemMecaProblem
import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver

class QuasiStaticMecaProblemBase(metaclass=abc.ABCMeta):
    def __init__(self,other=None):
        if other is None:
            self._staticProblem=None
            self.transientParams=dict()
            self.solutions=dict()
            self.timeSteps=np.array([])
        else:
            self._staticProblem=type(other._staticProblem)(other._staticProblem)
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

    def Preprocessing(self):
        self._staticProblem.Preprocessing()

    def BuildModel(self):
        self._staticProblem.BuildModel()

    def InitSolution(self):
        self.solutions={fieldType: [] for _,fieldType,_ in self._staticProblem.spacesVariables}

    def InitVariable(self,fieldType,field):
        self._staticProblem.InitVariable(fieldType,field)

    def SaveStaticSolution(self):
        for variable,fieldType,_ in self._staticProblem.spacesVariables:
            staticSolution=PhySolver.GetModelVariableValue(self._staticProblem.model,variable)
            self.solutions[fieldType].append(staticSolution)

    def GetSolution(self,fieldType):
        return self.solutions[fieldType]

    def SolveStaticProblem(self):
        state=self._staticProblem.RunProblem()
        if not state:
            raise Exception("Static solve has failed")

    def GetBasicCoordinates(self):
        return self._staticProblem.GetBasicCoordinates()

    def GetFeSpace(self,fieldType):
        return self._staticProblem.GetFeSpace(fieldType)

    def RunProblem(self):
        self.InitSolution()
        finalTime,dt=self.transientParams["time"],self.transientParams["timeStep"]
        step,currentTime=0,0.0

        timeSteps = []
        while currentTime<finalTime:
            currentTime=round(step*dt,10)
            print("Current time: ",currentTime)
            self.UpdateStaticProblem(currentTime)
            self.SolveStaticProblem()
            self.SaveStaticSolution()
            timeSteps.append(currentTime)
            step+=1

        self.timeSteps=np.array(timeSteps)

    def AtomicSolve(self):
        self.SolveStaticProblem()
        u = self._staticProblem.GetSolution(PFN.displacement)
        return u

    @abc.abstractmethod
    def UpdateStaticProblem(self,time):
        pass

    def __getattr__(self, name):
        def attrchecker(name):
            return lambda obj: hasattr(obj, name)

        possible = next(filter(attrchecker(name), [self._staticProblem]), None)
        if possible:
            return possible.__getattribute__(name)
        raise AttributeError("No attribute {!s} found".format(name))

    def ExportSolutionInCsv(self,filename,outputformat):
        feSpace,solutions=self.GetFeSpace(PFN.displacement),self.GetSolution(PFN.displacement)
        PhySolver.ExportTransientSolutionInCSV(filename,outputformat,feSpace,solutions)

    def ExportTimeStepsInCsv(self,filename):
        np.savetxt(filename, self.timeSteps, delimiter=',')

    def ExportSolution(self,filename,extension):
        if extension=="gmsh":
            self.ExportSolutionInGmsh(filename=filename)
        elif extension=="vtk":
            self.ExportSolutionInVTKSplitFiles(filename=filename)
        else:
            raise Exception("Extension "+str(extension)+" not recognized")

    def ExportSolutionInGmsh(self,filename,fieldType=PFN.displacement):
        feSpace,solutions=self.GetFeSpace(fieldType),self.GetSolution(fieldType)
        PhySolver.ExportTransientPrimalSolutionInGmsh(filename,feSpace,solutions)

    def ExportSolutionInVTKSplitFiles(self,filename,fieldType=PFN.displacement):
        feSpace,solutions=self.GetFeSpace(fieldType),self.GetSolution(fieldType)
        PhySolver.ExportTransientPrimalSolutionInVTKSplitFiles(filename,feSpace,solutions)

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
        PhySolver.ExportTransientPrimalDualSolutionInVtk(filename,feSpaces,solutions,self._staticProblem.refNumByRegion[contactRegion])

    def ExportSolutionWithMultipliersInGmsh(self,filename,contactRegion):
        feSpaces = self.GetFeSpace(PFN.displacement),self.GetFeSpace(PFN.contactMultiplier)
        solutions = self.GetSolution(PFN.displacement),self.GetSolution(PFN.contactMultiplier)
        PhySolver.ExportTransientPrimalDualSolutionInGmsh(filename,feSpaces,solutions,self._staticProblem.refNumByRegion[contactRegion])


class QuasiStaticRollingProblem(QuasiStaticMecaProblemBase):
    def __init__(self,other=None):
        super(QuasiStaticRollingProblem,self).__init__(other)
        if other is None:
            self._staticProblem=GetfemRollingWheelProblem()

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
        PhySolver.ExportTransientPrimalSolutionInVTK(filename, feSpace, [solutions[-1]])

    def UpdateStaticProblem(self,time):
        self.UpdateRollingCondition(time)

    def UpdateRollingCondition(self,time):
        tagname,roll=self._staticProblem.rolling
        rollingParams={k: roll[k] for k in set(list(roll.keys())) - set(["type"])}
        newRolling=PhySolver.ComputeDISRollingCondition(self._staticProblem.refNumByRegion[tagname],
                                                         self._staticProblem.model,
                                                         self.GetFeSpace(PFN.displacement),
                                                         rollingParams["theta_Rolling"],
                                                         rollingParams["d"],
                                                         time)
        PhySolver.SetModelVariableValue(model=self._staticProblem.model, variableName='rollingRHS', variableValue=newRolling)

    def __str__(self):
        s="Quasi-static rolling problem\n"
        for paramName,paramVal in self.transientParams.items():
            s+="\t"+str(paramName)+": "+str(paramVal)+"\n"
        s+=self._staticProblem.__str__()+"\n"
        return s


class QuasiStaticMecanicalProblem(QuasiStaticMecaProblemBase):
    def __init__(self,other=None):
        super(QuasiStaticMecanicalProblem,self).__init__(other)
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
                PhySolver.SetModelVariableValue(model=self._staticProblem.model,variableName='DirichletData'+str(dirichId),variableValue=newDirichlet)
            else:
                raise Exception("Not handled yet")

    def UpdateNeumannConditions(self,time):
        for neumId,neum in enumerate(self._staticProblem.neumann):
            if neum[1]["type"]=="StandardNeumann":
                newFx,newFy=time*neum[1]["fx"],time*neum[1]["fy"]
                PhySolver.SetModelVariableValue(model=self._staticProblem.model,variableName='NeumannData'+str(neumId),variableValue=[newFx,newFy])
            else:
                raise Exception("Not handled yet")

    def __str__(self):
        s="Quasi-static problem\n"
        s+="\t"+self.enforcedCondition+" piloted\n"
        for paramName,paramVal in self.transientParams.items():
            s+="\t"+str(paramName)+": "+str(paramVal)+"\n"
        s+=self._staticProblem.__str__()+"\n"
        return s