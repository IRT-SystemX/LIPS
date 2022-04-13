#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This file introduce a Getfem++ interface.
#The idea is to combine the individual features relying on Getfem++ "bricks" to build a physical problem

import numpy as np
import math
import os
from copy import copy,deepcopy

import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver
from lips.physical_simulator.GetfemSimulator.GetfemProblemBase import GetfemProblemBase
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
from lips.physical_simulator.GetfemSimulator.Utilitaries import ComputeNorm2

class ParamNotFound(Exception):
    pass

class GetfemMecaProblem(GetfemProblemBase):
    def __init__(self,other=None):
        """
        .. py:method:: __init__(other)
        :param GetfemMecaProblem other: another instance of this very class        

        Constructor of the class GetfemMecaProblem
        :ivar int dim: dimension of the problem (2 or 3)       
        :ivar list materials: material properties      
        :ivar list dirichlet: dirichlet conditions
        :ivar list neumann: neumann conditions
        :ivar list contact: contact conditions
        :ivar list source: source terms
        """
        fieldsICanSupply = [PFN.equivDirichletNodalForces,PFN.contactUnilateralMultiplier,PFN.contactMultiplier]
        self.auxiliaryFieldGeneration = {fieldname:False for fieldname in fieldsICanSupply}

        super(GetfemMecaProblem,self).__init__(other)
        self.internalInitParams=dict()
        if other is None:
            self.dim= 2
            self._materials = [] #[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
            self._dirichlet = [] #[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
            self._neumann = [] #[["HOLE_BOUND",{"type" : "RimRigidityNeumann", "Force":1.0E2}] ]
            self._contact = [] #[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
            self._sources = [] #[["ALL",{"type" : "Uniform","val":(0.0,0.0)}] ]
            self.problemCharacsByType = dict()
            self.nodalForces= []
        else:
            self._materials = deepcopy(other._materials)
            self._dirichlet = deepcopy(other._dirichlet)
            self._neumann = deepcopy(other._neumann)
            self._contact = deepcopy(other._contact)
            self._sources = deepcopy(other._sources)
            self.problemCharacsByType = deepcopy(other.problemCharacsByType)
            self.nodalForces= deepcopy(other.nodalForces)

    def GetFieldFromName(self,fieldName):
        if fieldName==PFN.equivDirichletNodalForces:
            return self.ComputeDirichletNodalForces(self.auxiliaryParams[fieldName]["boundary"])
        elif fieldName==PFN.contactUnilateralMultiplier or fieldName==PFN.contactMultiplier:
            return self.GetSolution(fieldName)
        elif fieldName==PFN.displacementNorm:
            displacement=self.GetSolution(PFN.displacement)
            return ComputeNorm2(displacement)
        else:
            raise Exception("Field "+fieldName+" is not available")

    def ComputeDirichletNodalForces(self,dirichBoundary):
        mim = self.IntegrMethod["standard"]
        mesh=self.mesh
        mfu=self.feSpaces[PFN.displacement]
        nodalForces=PhySolver.EquivalentDirichletNodalForces(model=self.model,mesh=mesh,mim=mim,mfDisp=mfu,dirichBoundary=dirichBoundary)
        return nodalForces

    def UpdateInternalParams(self,brick):
        """
        .. py:method:: UpdateInternalParams()

        Build internal physical params structure as the problem bricks are defined
        """
        if len(brick)>1:
            print("Warning: automatic internal params not reliable in this configuration! To be implemented.")
        for tagname,params in brick:
            #We only keep parameters with a numerical value
            internalParams={key:val for key,val in params.items() if isinstance(val, float) or isinstance(val,int)}
            self.internalInitParams.update(internalParams)

    @property
    def materials(self):
        return self._materials

    @materials.setter
    def materials(self, value):
        self.UpdateInternalParams(value)
        self._materials = value

    @property
    def dirichlet(self):
        return self._dirichlet

    @dirichlet.setter
    def dirichlet(self, value):
        self.UpdateInternalParams(value)
        self._dirichlet = value

    @property
    def neumann(self):
        return self._neumann

    @neumann.setter
    def neumann(self, value):
        self.UpdateInternalParams(value)
        self._neumann = value

    @property
    def contact(self):
        return self._contact

    @contact.setter
    def contact(self, value):
        self.UpdateInternalParams(value)
        self._contact = value

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        self.UpdateInternalParams(value)
        self._sources = value

    def ExportSolutionWithMultipliersInGmsh(self,filename):
        # Solution export
        if not self.contact:
            raise Exception("Can not export contact multiplier if there is no contact problem!")
        u,mult = self.GetSolution(PFN.displacement),self.GetSolution(PFN.contactMultiplier)
        mfu,mfmult = self.GetFeSpace(PFN.displacement),self.GetFeSpace(PFN.contactMultiplier)
        refIndices=self.GetFeSpace(PFN.contactMultiplier).basic_dof_on_region(self.refNumByRegion["CONTACT_BOUND"])
        extendedLambda=np.zeros(mfmult.nbdof())
        extendedLambda[refIndices]=-mult
        mfu.export_to_pos(filename, mfu, u, 'Displacement',mfmult,extendedLambda,"Lagrange multiplier")

    def ContactMultipliersRequired(self):
        useMultipliersByName={
               "NoFriction":True,
               "Inclined":True,
               "Plane":True,
               "PlanePenalized":False}

        useMultipliers=[useMultipliersByName[cont["type"]] for tagname,cont in self.contact]
        return False if all(not flag for flag in useMultipliers) else True

    def SetVariables(self,variables=None,multiplierVariable=None):
        """
        .. py:method:: SetVariables()

        Set variables and finite element spaces
        """
        self.spacesVariables = [("u", PFN.displacement, None)]
        self.feDef ={
               PFN.displacement: {"degree": 2, "dof": self.dim},
               "rim": {"degree": 1, "dof": self.dim},
               "obstacle": {"degree": 2, "dof": 1},
              }

        if self.contact and self.ContactMultipliersRequired():
            if multiplierVariable is not None:
                multVariable=multiplierVariable
            else:
                multVariable=("lambda",PFN.contactMultiplier,"CONTACT_BOUND") 
            self.spacesVariables.append(multVariable) 
            self.feDef[PFN.contactMultiplier]={"degree": 1, "dof": self.dim}

        if variables is not None:
            for varName,val in variables.items():
                self.feDef[varName]=val


    def BuildModel(self):
        """
        .. py:method:: BuildModel()

        Clean model and build constitutive brick one by one, sequentially
        """
        self.CleanProblemModel()
        self.materialsbricks=self.BuildBehaviourLaw()
        self.bodyForcesBrick=self.BuildBodyForces()
        self.dirichletBrick=self.BuildDirichletBC()
        self.neumannBrick=self.BuildNeumannBC()
        self.contactBrick=self.BuildContactBC()
        if self.nodalForces:
            self.nodalForcesBrick=self.BuildNodalForces()

    def DeleteModelBrick(self,brickType):
        """
        .. py:method:: DeleteModelBrick(brickType)

        Remove a type of brick from the model
        :param string brickType: type of brick to remove        
        """
        if brickType=="dirichlet":
            brickTodelete=self.dirichletBrick
        else:
            raise Exception("This brick can not be deleted for now")
        PhySolver.DeleteBoundaryCondition(self.model,brickTodelete)

    def BuildBehaviourLaw(self):
        """
        .. py:method:: BuildBehaviourLaw()

        Build behaviour law bricks
        """
        self.problemCharacsByType["Material"]=self.materials

        behaviourLawByName={"LinearElasticity":PhySolver.AddLinearElasticity,
                            "IncompressibleMooneyRivlin":PhySolver.AddIncompMooneyRivlin,
                            "SaintVenantKirchhoff":PhySolver.AddSaintVenantKirchhoff
                            }
        materialsbricks=[]
        for tagname,material in self.materials:
            materialLaw=material["law"]
            materialParams={k: material[k] for k in set(list(material.keys())) - set(["law"])}
            if tagname=="ALL":
                materialbrick=behaviourLawByName[materialLaw](self.model,self.IntegrMethod["standard"],materialParams)
            else:
                materialbrick=behaviourLawByName[materialLaw](self.model,self.IntegrMethod["standard"],materialParams,tagname)
            materialsbricks.append(materialbrick)
        return materialsbricks


    def BuildBodyForces(self):
        """
        .. py:method:: BuildBodyForces()

        Build body forces bricks
        """
        self.problemCharacsByType["Sources"]=self.sources

        sourceTypeByName={"Uniform":PhySolver.AddUniformSourceTerm,"Variable":PhySolver.AddVariableSourceTerm}
        bodyForcesBricks=[]
        for tagname,source in self.sources:
            sourceType=source["type"]
            sourceParams={k: source[k] for k in set(list(source.keys())) - set(["type"])}
            if tagname=="ALL":
                bodyForcesBrick=sourceTypeByName[sourceType](self.model,self.IntegrMethod["standard"],sourceParams)
            else:
                bodyForcesBrick=sourceTypeByName[sourceType](self.model,self.IntegrMethod["standard"],sourceParams,tagname)
            bodyForcesBricks.append(bodyForcesBrick)
        return bodyForcesBricks

    def BuildDirichletBC(self):
        """
        .. py:method:: BuildDirichletBC()

        Build dirichlet bricks
        """
        self.problemCharacsByType["Dirichlet"]=self.dirichlet
        dirichletByName={"scalar": PhySolver.AddDirichletCondition,
                         "vector": PhySolver.AddDirichletConditionVector,
                         "GlobalVector": PhySolver.AddDirichletConditionWithSimplification,
                         "rhs": PhySolver.AddDirichletConditionRHS,
                         "AnglePiloted": PhySolver.AddRollingCondition
                }

        dirichletBricks=[]
        for dirichId,(tagname,dirich) in enumerate(self.dirichlet):
            dirichType=dirich["type"]
            dirichParams={k: dirich[k] for k in set(list(dirich.keys())) - set(["type"])}
            if dirichType == 'AnglePiloted':
                dirichletBrick=dirichletByName[dirichType](self.refNumByRegion[tagname],dirichId, self.model, self.feSpaces[PFN.displacement],
                                                      self.IntegrMethod["standard"], dirichParams)
            else:
                dirichletBrick=dirichletByName[dirichType](self.refNumByRegion[tagname],dirichId, self.model,self.IntegrMethod["standard"],dirichParams)
            dirichletBricks.append(dirichletBrick)
        return dirichletBricks

    def BuildNeumannBC(self):
        """
        .. py:method:: BuildNeumannBC()

        Build neumann bricks
        """
        self.problemCharacsByType["Neumann"]=self.neumann
        neumannByName={"RimRigidityNeumann":PhySolver.AddRimRigidityNeumannCondition,
                       "StandardNeumann":PhySolver.AddNeumannCondition}

        neumannBricks=[]
        for neumannId,(tagname,neum) in enumerate(self.neumann):
            neumType=neum["type"]
            neumParams={k: neum[k] for k in set(list(neum.keys())) - set(["type"])}
            neumannBrick=neumannByName[neumType](self.refNumByRegion[tagname],neumannId,self.model,self.GetFeSpace("rim"),self.IntegrMethod["standard"],neumParams)
            neumannBricks.append(neumannBrick)
        return neumannBricks

    def BuildNodalForces(self):
        """
        .. py:method:: BuildNodalForces()

        Build nodal forces bricks
        """
        nodalForcesBricks=[]
        for tagname,nodalForce in self.nodalForces:
            nodalForcesBrick=PhySolver.AddExplicitRHS(self.model,'u',nodalForce)
            nodalForcesBricks.append(nodalForcesBrick)
        return nodalForcesBricks

    def BuildContactBC(self):
        """
        .. py:method:: BuildContactBC()

        Build contact bricks
        """
        self.problemCharacsByType["Contact"]=self.contact
        contactTypeByName={
               "NoFriction":PhySolver.AddUnilatContact,
               "Inclined":PhySolver.AddInclinedUnilatContactWithFric,
               "Plane":PhySolver.AddUnilatContactWithFric,
               "PlanePenalized":PhySolver.AddPenalizedUnilatContactWithFric
               }

        contactTypeArgByName={
               "NoFriction":(self.model,self.IntegrMethod["composite"]),
               "Inclined":(self.model,self.IntegrMethod["composite"]),
               "Plane":(self.model,self.IntegrMethod["composite"]),
               "PlanePenalized":(self.model,self.GetFeSpace("obstacle"),self.IntegrMethod["composite"])
               }

        contactBricks=[]
        for tagname,cont in self.contact:
            contType=cont["type"]
            contParams={k: cont[k] for k in set(list(cont.keys())) - set(["type"])}
            contactTypeArgs=(self.refNumByRegion[tagname],)+contactTypeArgByName[contType]+(contParams,)
            contactBrick=contactTypeByName[contType](*contactTypeArgs)
            contactBricks.append(contactBrick)
        return contactBricks

    def GetAllPhyComponents(self):
        """
        .. py:method:: GetAllComponents()

        Get all existing physical component of the problem
        """
        return [self.materials,self.sources,self.dirichlet,self.neumann,self.contact]

    def GetPhyParam(self,paramName):
        """
        .. py:method:: GetPhyParam()

        Get the value of an existing physical parameter
        :param string paramName: name of the parameter        
        """
        allComponents=self.GetAllPhyComponents()
        value=[params[paramName] for component in allComponents for tag,params in component  if paramName in params]
        if len(value)==1:
            return value[0]
        elif not len(value):
            raise ParamNotFound("parameter "+paramName+" not found")
        else:
            raise Exception("Ambiguous: several values available for "+paramName+" parameter, not implemented yet")

    def SetPhyParams(self,valueByParams):
        """
        .. py:method:: GetPhyParam()

        Set the value of existing physical parameters
        :param dict valueByParams: parameter value with respect to parameter name        
        """
        allComponents=self.GetAllPhyComponents()
        for paramName,paramVal in valueByParams.items():
            foundParams=0
            for component in allComponents:
                for tag,params in component:
                    if paramName in params:
                        params[paramName]=paramVal
                        foundParams+=1

            if not foundParams:
                raise ParamNotFound("parameter "+paramName+" not found")
            elif foundParams>1:
                raise Exception("Ambiguous: "+str(foundParams)+" occurences found for "+paramName+" parameter, not implemented yet")
            else:
                pass

    def ExportSolutionInFile(self,filename):
        """
        .. py:method:: ExportSolutionInFile(filename)

        Export in .csv format a field computed using Getfem++
        :param string filename: output file name         
        """
        self.ExportDisplacementInFile(filename)
        self.ExportAuxiliaryFieldsInFile(filename)

    def ExportDisplacementInFile(self,filename):
        fieldType=PFN.displacement
        filePath=filename.split(os.sep)
        filePath[-1]=fieldType+filePath[-1]
        newFileName=os.path.join(*filePath)
        feSpace,solution=self.feSpaces[fieldType],self.GetSolution(fieldType)
        PhySolver.ExportSolutionInCSV(newFileName,"VectorSolution",feSpace,solution)

    def ExportAuxiliaryFieldsInFile(self,filename):
        for fieldName,tag in self.auxiliaryFieldGeneration.items():
            if tag:
                filePath=filename.split(os.sep)
                filePath[-1]=fieldName+filePath[-1]
                newFileName=os.path.join(*filePath)
                if fieldName==PFN.contactUnilateralMultiplier or fieldName==PFN.contactMultiplier:
                    if len(self.contact)>1:
                        raise Exception("Several contact label case not handled yet!")
                    tagname,_ =self.contact[0]
                    feSpace,solution=self.GetFeSpace(fieldName),self.GetAuxiliaryField(fieldName=fieldName)
                    if fieldName==PFN.contactUnilateralMultiplier:
                        outputFormat="ScalarField"
                    else:
                        outputFormat="VectorSolution"                
                    PhySolver.ExportMultipliersInCSV(newFileName,outputFormat,feSpace,solution,self.refNumByRegion[tagname])
                elif fieldName==PFN.displacementNorm:
                    feSpace,solution=self.GetFeSpace(PFN.displacement),self.GetAuxiliaryField(fieldName=fieldName)
                    PhySolver.ExportSolutionInCSV(newFileName,'ScalarField',feSpace,solution)
                elif fieldName==PFN.equivDirichletNodalForces:
                    feSpace,fieldVal=self.GetFeSpace(PFN.displacement),self.GetAuxiliaryField(fieldName=fieldName)
                    PhySolver.ExportNodalStressFieldInCSV(newFileName,"VectorSolution",feSpace,fieldVal)
                else:
                    raise Exception("Field "+str(fieldName)+" not available") 

    def __str__(self):
        s=super().__str__()+"\n"
        for characName,characValues in self.problemCharacsByType.items():
            if characValues:
                s+=str(characName)+"\n"
                for _,characValue in characValues:
                    for paramName,paramVal in characValue.items():
                        s+="\t"+str(paramName)+": "+str(paramVal)+"\n"
                    s+="\n"
        return s


class GetfemRollingWheelProblem(GetfemMecaProblem):
    def __init__(self,other=None):
        """
        .. py:method:: __init__(other)
        :param GetfemRollingWheelProblem other: another instance of this very class        

        Constructor of the class GetfemRollingWheelProblem
        :ivar list rolling: rolling conditions
        """
        super(GetfemRollingWheelProblem,self).__init__(other)

        if other is None:
            self._rolling = [] #[["HOLE_BOUND",{"type" : "AnglePiloted","theta_Rolling":math.pi/8}]]
        else:
            self._rolling = deepcopy(other.rolling)

    @property
    def rolling(self):
        return self._rolling

    @rolling.setter
    def rolling(self, value):
        self.UpdateInternalParams([value])
        self._rolling = value

    def BuildRollingBC(self):
        """
        .. py:method:: BuildRollingBC()

        Build rolling bricks
        """
        self.problemCharacsByType["Rolling"]=[self.rolling]

        rollingTypeByName = {"AnglePiloted": PhySolver.AddRollingCondition,
                             "DIS_Rolling": PhySolver.AddDisplacementImposedRollingCondition,
                             "FORC_Rolling": PhySolver.AddForceImposedRollingCondition }

        tagname,roll=self.rolling
        rollingType=roll["type"]
        rollingParams={k: roll[k] for k in set(list(roll.keys())) - set(["type"])}
        rollingBrick=rollingTypeByName[rollingType](self.refNumByRegion[tagname],self.model,self.GetFeSpace(PFN.displacement),self.IntegrMethod["standard"],rollingParams)

        return rollingBrick

    def GetAllPhyComponents(self):
        """
        .. py:method:: GetAllComponents()

        Get all existing physical component of the problem
        """
        return [self.materials,self.sources,self.dirichlet,self.neumann,self.contact,[self.rolling]]

    def GetRollingConditionType(self):
        if not self.rolling:
            raise Exception("There is no rolling condition in this problem!")
        _,roll=self.rolling
        return roll["type"]

    def BuildModel(self):
        """
        .. py:method:: BuildModel()

        Clean model and build constitutive brick one by one, sequentially
        """
        super().BuildModel()
        rollingType=self.GetRollingConditionType()
        if rollingType=="AnglePiloted":
            pass
        elif rollingType=="DIS_Rolling" or rollingType=="FORC_Rolling":
            self.rollingBrick=self.BuildRollingBC()
        else:
            raise Exception("Do not know how to handle"+str(self.rollingType)+" rolling condition")

    def RunProblem(self,noisySolve=True):
        """
        .. py:method:: RunProblem()

        Solve physical problem
        """
        rollingType=self.GetRollingConditionType()
        if rollingType=="AnglePiloted":
            super().RunProblem()
            self.DeleteModelBrickFromId(self.dirichletBrick)
            #Define rolling brick
            self.rollingBrick=self.BuildRollingBC()
            return super().RunProblem()
        elif rollingType=="DIS_Rolling":
            return super().RunProblem()
        elif rollingType=="FORC_Rolling":
            solve=PhySolver.TwoStepsRollingSolve(self)
            self.SaveSolutions()
            return solve
        else:
            raise Exception("Do not know how to handle"+str(rollingType)+" rolling condition")