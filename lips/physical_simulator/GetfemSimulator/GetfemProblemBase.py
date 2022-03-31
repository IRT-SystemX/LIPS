#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import copy,deepcopy

import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver

class FieldNotFound(Exception):
    pass

class GetfemProblemBase:
    """
    .. py:class:: GetfemProblemBase

    Base class for Getfem Problems
    """
    def __init__(self,other=None):
        """
        .. py:method:: __init__(other)
        :param GetfemProblemBase other: instance of this very class        

        Constructor of the class GetfemProblemBase
        :ivar int dim: dimension of the problem (2 or 3)       
        :ivar Model model: store the constitutive bricks of the problem  in Getfem       
        :ivar MeshIm by name IntegrMethod: Integration methods interfaced with Getfem
        :ivar MeshFem by variable feSpaces: Finite element description with respect to each variable
        :ivar list variables: name, finite element space, spatial support tag in mesh
        :ivar dict feDef: finite element description with respect to unknown
        :ivar list spacesVariables: variable name,unknown, spatial support tagname
        """
        if other is None:
            self.dim = None
            self.model = None
            self.mesh = None
            self.refNumByRegion=dict()
            self.IntegrMethod = dict()
            self.feSpaces = dict()
            self.variables = []
            self.feDef = dict()
            self.spacesVariables = []
            self.solutions=dict()
            self.auxiliaryFieldGeneration=dict()
            self.auxiliaryField=dict()
            self.auxiliaryParams=dict()
            self.max_residual=1e-6
            self.max_iter=100
        else:
            self.dim = other.dim
            self.model = other.model
            self.mesh = other.mesh
            self.refNumByRegion=other.refNumByRegion
            self.IntegrMethod = other.IntegrMethod
            self.feSpaces = other.feSpaces
            self.variables = other.variables
            self.feDef = other.feDef
            self.spacesVariables = other.spacesVariables
            self.solutions=deepcopy(other.solutions)
            self.auxiliaryFieldGeneration=deepcopy(other.auxiliaryFieldGeneration)
            self.auxiliaryField=deepcopy(other.auxiliaryField)
            self.auxiliaryParams=deepcopy(other.auxiliaryParams)
            self.max_residual=other.max_residual
            self.max_iter=other.max_iter


    def SetFeSpaces(self):
        """
        .. py:method:: SetFeSpaces()

        Define finite element spaces
        """
        for variable,space in self.feDef.items():
            self.feSpaces[variable]=PhySolver.DefineFESpaces(mesh=self.mesh,elements_degree=space["degree"],dof=space["dof"])

    def GetFeSpace(self,fieldType):
        """
        .. py:method:: GetFeSpace()

        Retrieve finite element space associated to a field
        """
        try:
            feSpace=self.feSpaces[fieldType]
        except KeyError:
            availfeSpaces=', '.join("{}".format(k) for k in self.feSpaces.keys())
            raise FieldNotFound("Finite element space for field "+fieldType+" not available. Available fields:\n"+availfeSpaces)

        return feSpace


    def SetIntegrationMethods(self):
        """
        .. py:method:: SetIntegrationMethods()

        Define integration methods
        """
        self.IntegrMethod={
                "standard":PhySolver.DefineIntegrationMethodsByOrder(mesh=self.mesh,order=4),
                "composite":PhySolver.DefineCompositeIntegrationMethodsByName(mesh=self.mesh, name='IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(4),2)')
                           }

    def InitModel(self):
        """
        .. py:method:: InitModel()

        Init Getfem model, create variables related to unknown in model
        """
        self.model=PhySolver.DefineModel()
        for var,space,bound in self.spacesVariables:
            if bound is not None:
                boundRef=self.refNumByRegion[bound]
            else:
                boundRef=bound
            PhySolver.AddVariable(self.model,var,self.feSpaces[space],boundRef)
            self.variables.append((var,self.feSpaces[space],boundRef))

    def InitVariable(self,fieldType,fieldValue):
        """
        .. py:method:: InitModelWithSolution(solution)

        Init model variables with user defined vector
        """
        varNameByvarType={PFN.displacement:"u",PFN.contactMultiplier:"lambda"}
        modelVariableName=varNameByvarType[fieldType]
        PhySolver.SetModelVariableValue(self.model,modelVariableName,fieldValue)

    def Preprocessing(self,variables=None,multiplierVariable=None):
        """
        .. py:method:: Preprocessing()

        Must be done once before building problem, define variables, finite element spaces, integrations methods and model
        """
        self.SetVariables(variables=variables,multiplierVariable=multiplierVariable)
        self.SetFeSpaces()
        self.SetIntegrationMethods()        
        self.InitModel()

    def RunProblem(self,noisySolve=True):
        """
        .. py:method:: RunProblem()

        Solve physical problem
        """
        print('Solve problem with ', PhySolver.GetNbDof(self.model), ' dofs')
        solverState=PhySolver.Solve(model=self.model,max_iter=self.max_iter,max_residual=self.max_residual,noisiness=noisySolve)        
        self.SaveSolutions()
        self.SaveAuxiliaryFields()
        return solverState

    def SaveSolutions(self):
        """
        .. py:method:: SaveSolutions()

        Retrieve solutions from solved problem
        """
        for variable,fieldType,_ in self.spacesVariables:
            self.solutions[fieldType]=PhySolver.GetModelVariableValue(self.model,variable)

    def AssembleProblem(self):
        """
        .. py:method:: AssembleProblem()

        Assemble physical problem
        """
        PhySolver.AssembleProblem(self.model)

    def ExtractFullAssembledSystem(self):
        """
        .. py:method:: AssembleProblem()

        Assemble physical problem
        """
        return PhySolver.ExtractFullAssembledSystem(self.model)

    def ExtractRHS(self):
        return PhySolver.ExtractRHS(self.model)

    def AddExplicitRHSForField(self,fieldName,explicitRHS):
        variableModelName = self.GetModelVariableName(fieldName)
        PhySolver.AddExplicitRHS(self.model,variableModelName,explicitRHS)

    def AddExplicitConstraintForFieldWithMult(self,fieldName,multVariableName,matConstraint,rhsConstraint):
        variableModelName = self.GetModelVariableName(fieldName)
        PhySolver.AddConstraintWithMultipliers(self.model,variableModelName,multVariableName,matConstraint,rhsConstraint)

    def GetVariableValue(self,fieldName):
        variableModelName = self.GetModelVariableName(fieldName)
        return PhySolver.GetModelVariableValue(self.model,variableModelName)

    def GetVariableDofInterval(self,fieldName):
        variableModelName=self.GetModelVariableName(fieldName)
        varStart,varEnd = PhySolver.GetVariableDofInterval(self.model,variableModelName)
        return varStart,varEnd

    def GetModelVariableName(self,fieldName):
        try:
            variableModelName=PhySolver.modelVarByPhyField[fieldName]
        except KeyError:
            raise Exception("The field "+fieldName+" was not found within the model.")
        return variableModelName

    def AddFixedSizeVariable(self,variableName,variableSize):
        PhySolver.AddFixedSizeVariable(self.model,variableName,variableSize)

    def GetSolution(self,fieldType):
        """
        .. py:method:: GetSolution()

        Retrieve solutions associated to a field
        """
        try:
            solution=self.solutions[fieldType]
        except KeyError:
            availFields=', '.join("{}".format(k) for k in solutions.keys())
            raise FieldNotFound("Solution for field "+fieldType+" not available. Available fields:\n"+availFields)

        return solution

    def SetAuxiliaryField(self,fieldName,tag=True,params=None):
        self.auxiliaryFieldGeneration[fieldName]=tag
        if tag and params is not None:
            self.auxiliaryParams[fieldName]=params

    def GetAuxiliaryField(self,fieldName):
        try:
            auxiliaryField=self.auxiliaryField[fieldName]
        except KeyError:
            availFields=', '.join("{}".format(k) for k in auxiliaryField.keys())
            raise FieldNotFound("auxiliary field for "+fieldType+" not available. Available auxiliary fields:\n"+availFields)
        return auxiliaryField

    def SaveAuxiliaryFields(self):
        """
        .. py:method:: SaveFields()

        Retrieve fields from solved problem
        """
        for fieldName,tag in self.auxiliaryFieldGeneration.items():
            if tag:
                self.auxiliaryField[fieldName]=self.GetFieldFromName(fieldName)

    def RestoreVariables(self):
        """
        .. py:method:: RestoreVariables()

        Recreate original variables in model (to use if they were deleted)
        """
        for var,mfvar,bound in self.variables:
            PhySolver.AddVariable(self.model,var,mfvar,bound)

    def CleanProblemModel(self):
        """
        .. py:method:: CleanProblemModel()

        Delete every bricks and variables in the model
        """
        PhySolver.CleanModel(self.model)
        self.RestoreVariables()

    def DeleteModelBrickFromId(self,brickId):
        PhySolver.DeleteBoundaryCondition(self.model,brickId)

    def DeleteModelVariable(self,variableName):
        PhySolver.DeleteVariable(self.model,variableName)

    def ExportFieldFromFileInGmsh(self,fileName,outputformat,solutionName):
        if outputformat == 'VectorSolution':
            ux, uy = np.loadtxt(fileName, delimiter=",", skiprows=1, usecols=(0, 1),unpack=True)
            field=np.array([[uxi,uyi] for uxi,uyi in zip(ux,uy)]).flatten()
            dofperNodes=2
        elif outputformat == 'ScalarSolution':
            field = np.loadtxt(fileName,delimiter=",",skiprows=1,usecols=(0),unpack=True)
            dofperNodes=1
        else:
            raise Exception("Format not recognized ",outputformat)
        outputName=fileName.split('.')[0]
        self.ExportFieldInGmshWithFormat(filename=outputName+"ForGmsh.pos",field=field,elements_degree=2,dofpernodes=dofperNodes,fieldName=solutionName)

    def ExportFieldInGmsh(self,filename,field,fieldName,fieldType=PFN.displacement):
        """
        .. py:method:: ExportFieldInGmsh(filename,fieldType,field,fieldName)

        Export in .msh format a field not computed using Getfem++
        :param string filename: output file name        
        :param array field: to be exported field value
        :param string fieldName: to be exported field name
        :param string fieldType: Getfem unknown related to field     
        """
        feSpace,solutions=self.feSpaces[fieldType],self.GetSolution(fieldType)
        dummyMffield=PhySolver.DefineFESpaces(mesh=self.mesh,elements_degree=2,dof=1)
        PhySolver.ExportFieldInGmsh(filename,feSpace,solutions,dummyMffield,field,fieldName)

    def ExportFieldInGmshWithFormat(self,filename,field,dofpernodes,fieldName,elements_degree=2):
        """
        .. py:method:: ExportFieldInGmsh(filename,fieldType,field,fieldName)

        Export in .msh format a field not computed using Getfem++
        :param string filename: output file name        
        :param array field: to be exported field value
        :param string fieldName: to be exported field name
        """
        dummyfeSpace=PhySolver.DefineFESpaces(mesh=self.mesh,elements_degree=elements_degree,dof=dofpernodes)
        PhySolver.ExportSingleFieldInGmsh(filename,dummyfeSpace,field,fieldName)

    def ExportSolution(self,filename,extension,fieldType=PFN.displacement):
        feSpace,solutions=self.GetFeSpace(fieldType),self.GetSolution(fieldType)
        if extension=="gmsh":
            self.ExportSolutionInGmsh(filename,fieldType)
        elif extension=="vtk":
            self.ExportSolutionInVTK(filename,fieldType)
        else:
            raise Exception("Extension "+str(extension)+" not available")

    def ExportSolutionInGmsh(self,filename,fieldType=PFN.displacement):
        """
        .. py:method:: ExportSolutionInGmsh(filename,fieldType)

        Export in .msh format a field computed using Getfem++
        :param string filename: output file name        
        :param string fieldType: Getfem unknown related to field     
        """
        feSpace,solutions=self.feSpaces[fieldType],self.GetSolution(fieldType)
        PhySolver.ExportPrimalSolutionInGmsh(filename,feSpace,solutions)

    def ExportSolutionInVTK(self,filename,fieldType=PFN.displacement):
        """
        .. py:method:: ExportSolutionInVTK(filename,fieldType)

        Export in .msh format a field computed using Getfem++
        :param string filename: output file name        
        :param string fieldType: Getfem unknown related to field     
        """
        feSpace,solutions=self.GetFeSpace(fieldType),self.GetSolution(fieldType)
        PhySolver.ExportPrimalSolutionInVTK(filename,feSpace,solutions)
  

    def GetBasicDof(self,fieldType=PFN.displacement,regionTag=None,dofIds=None):
        """
        .. py:method:: GetBasicDof(fieldType)  
        """
        feSpace=self.feSpaces[fieldType]
        return PhySolver.GetBasicDof(mfu=feSpace,regionTag=regionTag,dofIds=dofIds)

    def GetSolutionAsField(self,fieldType=PFN.displacement):
        """
        .. py:method:: GetSolutionAsField(fieldType)

        Extract solution as a field computed using Getfem++       
        :param string fieldType: Getfem unknown related to field     
        """
        feSpace,solutions=self.feSpaces[fieldType],self.GetSolution(fieldType)
        return PhySolver.GetSolutionAsField(feSpace,solutions)

    def GetBasicCoordinates(self,fieldType=PFN.displacement):
        """
        .. py:method:: GetBasicCoordinates(fieldType)

        Extract dof coordinates       
        :param string fieldType: Getfem unknown related to field     
        """
        feSpace=self.feSpaces[fieldType]
        return PhySolver.GetBasicCoordinates(feSpace)

    def ComputeSpatialErrorIndicator(self,field,errorType):
        if errorType=="L2":
            return PhySolver.ComputeL2Norm(self.feSpaces[PFN.displacement],field,self.IntegrMethod["standard"])
        elif errorType=="H1":
            return PhySolver.ComputeH1Norm(self.feSpaces[PFN.displacement],field,self.IntegrMethod["standard"])
        elif errorType=="L_inf":
            return np.max(field)
        else:
            raise Exception("errorType "+errorType+" not available")

    def __str__(self):
        print("Model description")
        PhySolver.PrintModel(self.model)

        s="Dimension: "+str(self.dim)+"\n"
        s+="Variables: \n"
        for var,space,bound in self.spacesVariables:
            if bound is None:
                s+="\t Name: "+var+"\n"
            else:
                s+="\t Name "+var+" defined on "+bound+"\n"
            s+="\t Element degree: "+str(self.feDef[space]["degree"])
            s+="\t Dof per node: "+str(self.feDef[space]["dof"])+"\n"
        return s
