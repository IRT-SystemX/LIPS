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

#################################Test#################################
import numpy.testing as npt

def CheckIntegrity_solutionContactComputed():
    print("Testing linear elasticity with regular contact")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]

    youngVal=WPM.GetPhyParam("young")
    npt.assert_equal(youngVal,21E6)
    poissonVal=WPM.GetPhyParam("poisson")
    npt.assert_equal(poissonVal,0.3)
    dispVal=WPM.GetPhyParam("Disp_Amplitude")
    npt.assert_equal(dispVal,6)

    npt.assert_raises(ParamNotFound, WPM.GetPhyParam, "toto")

    WPM.SetPhyParams({"young":10E6,"poisson":0.25})
    youngVal=WPM.GetPhyParam("young")
    npt.assert_equal(youngVal,10E6)
    poissonVal=WPM.GetPhyParam("poisson")
    npt.assert_equal(poissonVal,0.25)

    npt.assert_raises(ParamNotFound, WPM.SetPhyParams, {"toto":3})
    return "ok"

def CheckIntegrity_solutionLinearElasticity():
    print("Testing linear elasticity case")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":-5e3}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":0, "Disp_Angle":0}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)

    #Test solution
    solution=WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(solution), -0.00563, atol=1e-5)

    #Build the problem and run again
    WPM.BuildModel()
    WPM.RunProblem()
    solution=WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(solution), -0.00563, atol=1e-5)
    return "ok"

def CheckIntegrity_solutionContactComputed():
    print("Testing linear elasticity with regular contact")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)
    WPM.ExportSolutionWithMultipliersInGmsh("ContactSolution.pos")

    disp = WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(disp),-2.6986,atol=1e-3)
    mult = WPM.GetSolution(PFN.contactMultiplier)
    npt.assert_allclose(np.mean(mult),1752126.96331,atol=1e0)
    return "ok"

def CheckIntegrity_solutionContactInclinedFoundation():
    print("Testing linear elasticity with unilateral contact and friction case with oriented foundation")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Inclined","Found_angle":-math.pi/2,"gap":0.0,"fricCoeff":0.9}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)

    #Test solution
    disp = WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(disp),-2.4643,atol=1e-3)
    mult = WPM.GetSolution(PFN.contactMultiplier)
    npt.assert_allclose(np.mean(mult),3049873.48013,atol=1e0)

    print("Testing clean model with new params and run again")
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":4, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Inclined","Found_angle":-3*math.pi/8,"gap":0.0,"fricCoeff":0.0}] ]
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)
    disp = WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(disp),-1.849,atol=1e-3)

    print("Testing clean model with new params and run again")
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":2, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Inclined","Found_angle":-math.pi/4,"gap":0.0,"fricCoeff":0.0}] ]
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)
    disp = WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(disp),-1.000,atol=1e-3)
    return "ok"


def CheckIntegrity_VariableForcesolutionComputed():
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    WPM.Preprocessing()

    print("Testing linear elasticity with unilateral contact and friction case with variable force")
    source_expressions=["sin(X),sin(Y)",
                        "X*Y,cos(X+Y)",
                        "exp(X),tanh(2*Y)"
                        ]

    meanSolution=[-2.6986,-2.6987,-2.6574]

    for id_source,source_expression in enumerate(source_expressions):
        WPM.sources=[["ALL",{"type" : "Variable","source_term":source_expression}] ]
        WPM.BuildModel()
        state=WPM.RunProblem()
        print("Solver state: ",state)
        solution=WPM.GetSolution(PFN.displacement)
        npt.assert_allclose(np.mean(solution),meanSolution[id_source],atol=1e-3)
    return "ok"


def CheckIntegrity_NeumannForceSolutionComputed():
    print("Testing nonlinear elasticity with unilateral contact and friction case with neumann condition")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8., 15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions, meshSize=1, RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials = [["ALL", {"law":"IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2":1} ]]
    WPM.neumann = [["HOLE_BOUND", {"type": "RimRigidityNeumann", "Force": 1.0E2}]]
    WPM.sources = [["ALL", {"type": "Uniform", "source_x": 0.0, "source_y": 0.0}]]
    WPM.contact =[["CONTACT_BOUND", {"type": "Plane", "gap": 0.0, "fricCoeff": 0.0}]]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)
    WPM.ExportSolution(filename="QuasiStaticRolling.pos", extension="gmsh")

    WPMBis=type(WPM)()

    # Test solution
    solution=WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(solution),-1.8150,atol=1e-3)
    contactMultipliers=WPM.GetSolution(PFN.contactMultiplier)
    npt.assert_allclose(np.mean(contactMultipliers),1.0838,atol=1e-2)
    return "ok"


def CheckIntegrity_RollingContactLinearAnglePiloted():
    print("Testing linear elasticity with unilateral contact and friction case with AnglePiloted rolling condition")
    WPM=GetfemRollingWheelProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    WPM.rolling=["HOLE_BOUND",{"type" : "AnglePiloted","theta_Rolling":math.pi/8}]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)

    #Test solution
    firstSolution=WPM.GetSolution(PFN.displacement)
    npt.assert_allclose(np.mean(firstSolution),-3.064,atol=1e-2)
    firstContactMultipliers=WPM.GetSolution(PFN.contactMultiplier)
    npt.assert_allclose(np.mean(firstContactMultipliers),779548.20,atol=1e0)

    #Copy the problem, build the model and run again
    WPMCopy=type(WPM)(WPM)
    WPMCopy.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)

    solution=WPMCopy.GetSolution(PFN.displacement)
    np.testing.assert_array_almost_equal(solution, firstSolution)
    contactMultipliers=WPMCopy.GetSolution(PFN.contactMultiplier)
    np.testing.assert_array_almost_equal(contactMultipliers,firstContactMultipliers)

    #Check modifying physical property of new problem does not affect original
    WPMCopy.materials=[["ALL", {"law":"LinearElasticity","young":30E6,"poisson":0.25} ]]
    WPMCopy.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":-3.0}] ]
    WPMCopy.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":3, "Disp_Angle":-math.pi/4}] ]
    WPMCopy.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.5}] ]
    WPMCopy.rolling=["HOLE_BOUND",{"type" : "AnglePiloted","theta_Rolling":math.pi/4}]

    WPMProps=[WPM.materials,WPM.sources,WPM.dirichlet,WPM.contact,WPM.rolling]
    WPMCopyProps=[WPMCopy.materials,WPMCopy.sources,WPMCopy.dirichlet,WPMCopy.contact,WPMCopy.rolling]
    for oriProp,copyProps in zip(WPMProps,WPMCopyProps):
        with npt.assert_raises(AssertionError):
            npt.assert_array_equal(oriProp,copyProps)
    return "ok"

def CheckIntegrity_RollingContactDIS_Rolling():
    print("Testing linear elasticity with unilateral contact and friction case with DIS_Rolling rolling condition")
    WPM=GetfemRollingWheelProblem()
    wheelDimensions=(8., 15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMeshRolling(wheelDimensions=wheelDimensions,meshSize=2,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials = [["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1}]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.6}] ]
    WPM.rolling=["HOLE_BOUND",{"type" : "DIS_Rolling", "theta_Rolling":150., 'd': 1.,'currentTime':0.0}]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)
    return "ok"

def CheckIntegrity_RollingContactFORC_Rolling():
    print("Testing linear elasticity with unilateral contact and friction case with FORC_Rolling rolling condition")
    WPM=GetfemRollingWheelProblem()
    wheelDimensions=(8., 15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMeshRolling(wheelDimensions=wheelDimensions,meshSize=2,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials = [["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1}]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.5}] ]
    fx=-1.0/(16*np.pi)
    fy=6.0/(16*np.pi)
    WPM.rolling=["HOLE_BOUND",{"type" : "FORC_Rolling", "forc_x":fx,"forc_y":fy,'currentTime':0.0001}]

    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()

    print("Solver state: ",state)
    # import getfem as gf
    # mim=WPM.IntegrMethod["composite"]
    # mfu=WPM.GetFeSpace(PFN.contactMultiplier)
    # val=gf.asm_generic(mim,1,"lambda.Test_t",refNumByRegion["CONTACT_BOUND"],WPM.model,"t")
    # print("toto: ",val)
    # print(WPM.model.variable("lambda"))
    #WPM.ExportSolutionWithMultipliersInGmsh("TestStaticRolling.pos")
    return "ok"


def CheckIntegrity_InitWithSolutionLinearElast():
    print("Testing initialization of problem with user defined solution")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)

    disp = WPM.GetSolution(PFN.displacement)

    WPMCopy=GetfemMecaProblem(WPM)
    WPMCopy.BuildModel()
    WPMCopy.InitVariable(PFN.displacement,disp)
    state=WPM.RunProblem()
    print("Solver state: ",state)

    disp2 =WPMCopy.GetSolution(PFN.displacement)
    errorDisp=np.nan_to_num(np.abs(disp2 - disp) / np.abs(disp) )

    npt.assert_array_less(np.max(errorDisp),1e-3)
    return "ok"

def CheckIntegrity_InitWithSolutionContact():
    print("Testing initialization of problem with user defined solution")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)
    disp = WPM.GetSolution(PFN.displacement)
    mult = WPM.GetSolution(PFN.contactMultiplier)

    WPMCopy=GetfemMecaProblem(WPM)
    WPMCopy.BuildModel()
    WPMCopy.InitVariable(PFN.displacement,disp)
    WPMCopy.InitVariable(PFN.contactMultiplier,mult)
    state=WPMCopy.RunProblem()
    print("Solver state: ",state)
    disp2 =WPMCopy.GetSolution(PFN.displacement)
    mult2 =WPMCopy.GetSolution(PFN.contactMultiplier)
    errorDisp=np.nan_to_num(np.abs(disp2 - disp) / np.abs(disp) )
    errorMult=np.nan_to_num(np.abs(mult2 - mult) / np.abs(mult) )

    npt.assert_array_less(np.max(errorDisp),1e-2)
    npt.assert_array_less(np.max(errorMult),1e-2)
    return "ok"

def CheckIntegrity_InitWithSolutionContactNoMultiplier():
    print("Testing initialization of problem with user defined solution")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=PhySolver.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":1.,"poisson":0.3} ]]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "PlanePenalized","fricCoeff":0.4}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    WPM.RunProblem()
    disp = WPM.GetSolution(PFN.displacement)

    WPMCopy=GetfemMecaProblem(WPM)
    WPMCopy.BuildModel()
    WPMCopy.InitVariable(PFN.displacement,disp)
    WPMCopy.RunProblem()
    disp2 =WPMCopy.GetSolution(PFN.displacement)
    errorDisp=np.nan_to_num(np.abs(disp2 - disp) / np.abs(disp) )

    npt.assert_array_less(np.percentile(errorDisp,99),1e-2)
    return "ok"

def CheckIntegrity_BeamNodalForcesFromDirichlet():
    BeamProblem=GetfemMecaProblem()
    BeamProblem.mesh,BeamProblem.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    BeamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}],["RIGHT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    BeamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    BeamProblem.Preprocessing(variables=femVariables)
    BeamProblem.SetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces,params={"boundary":BeamProblem.refNumByRegion["LEFT"]})
    BeamProblem.BuildModel()
    BeamProblem.RunProblem()
    sol0=BeamProblem.GetSolution(fieldType=PFN.displacement)
    nodalForces=BeamProblem.GetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces)

    BeamProblem2=GetfemMecaProblem()
    BeamProblem2.mesh,BeamProblem2.refNumByRegion=BeamProblem.mesh,BeamProblem.refNumByRegion
    BeamProblem2.materials=BeamProblem.materials
    BeamProblem2.dirichlet=[["RIGHT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    BeamProblem2.neumann=BeamProblem.neumann
    BeamProblem2.nodalForces=[["ALL",nodalForces]]
    BeamProblem2.Preprocessing(variables=femVariables)
    BeamProblem2.BuildModel()
    BeamProblem2.RunProblem()
    sol1=BeamProblem2.GetSolution(fieldType=PFN.displacement)
    relaError=np.linalg.norm(sol1-sol0)/np.linalg.norm(sol0)
    absError=np.linalg.norm(sol1-sol0)
    npt.assert_array_less(np.abs(relaError), 1e-6, "Relative error too large")
    npt.assert_array_less(np.abs(absError), 1e-6, "Absolute error too large")
    return "ok"

def CheckIntegrity_BeamNodalForcesFromNeumann():
    BeamNeumannProblem=GetfemMecaProblem()
    BeamNeumannProblem.mesh,BeamNeumannProblem.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamNeumannProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    BeamNeumannProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}]]
    BeamNeumannProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    BeamNeumannProblem.Preprocessing()
    BeamNeumannProblem.BuildModel()
    BeamNeumannProblem.RunProblem()
    sol0=BeamNeumannProblem.GetSolution(fieldType=PFN.displacement)

    BeamNeumannProblem.DeleteModelBrick("dirichlet")
    BeamNeumannProblem.AssembleProblem()
    rhs=BeamNeumannProblem.ExtractRHS()

    BeamNeumannProblem2=GetfemMecaProblem()
    BeamNeumannProblem2.mesh,BeamNeumannProblem2.refNumByRegion=BeamNeumannProblem.mesh,BeamNeumannProblem.refNumByRegion
    BeamNeumannProblem2.materials=BeamNeumannProblem.materials
    BeamNeumannProblem2.dirichlet=BeamNeumannProblem.dirichlet
    BeamNeumannProblem2.nodalForces=[["ALL",rhs]]
    BeamNeumannProblem2.Preprocessing()
    BeamNeumannProblem2.BuildModel()
    BeamNeumannProblem2.RunProblem()
    sol1=BeamNeumannProblem2.GetSolution(fieldType=PFN.displacement)
    relaError=np.linalg.norm(sol1-sol0)/np.linalg.norm(sol0)
    absError=np.linalg.norm(sol1-sol0)
    npt.assert_array_less(np.abs(relaError), 1e-6, "Relative error too large")
    npt.assert_array_less(np.abs(absError), 1e-6, "Absolute error too large")
    return "ok"

def CheckIntegrity_BeamSolProjection():
    BeamSolProj=GetfemMecaProblem()
    coarseMesh,numByRegion=PhySolver.GenerateSimpleMesh(meshSize=32.0)
    BeamSolProj.mesh,BeamSolProj.refNumByRegion=coarseMesh,numByRegion
    BeamSolProj.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    BeamSolProj.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}]]
    BeamSolProj.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    BeamSolProj.Preprocessing()
    BeamSolProj.BuildModel()
    BeamSolProj.RunProblem()
    sol0=BeamSolProj.GetSolution(fieldType=PFN.displacement)

    refinedMesh,numByRegion=PhySolver.GenerateSimpleMesh(meshSize=64.0)
    projSol=PhySolver.ProjectSolOnMesh(coarseSol=sol0,solDegree=2,coarseMesh=coarseMesh,refMesh=refinedMesh)

    BeamSolProj2=GetfemMecaProblem()
    BeamSolProj2.mesh,BeamSolProj2.refNumByRegion=refinedMesh,numByRegion
    BeamSolProj2.materials=BeamSolProj.materials
    BeamSolProj2.dirichlet=BeamSolProj.dirichlet
    BeamSolProj2.neumann=BeamSolProj.neumann
    BeamSolProj2.Preprocessing()
    BeamSolProj2.BuildModel()
    BeamSolProj2.RunProblem()
    sol1=BeamSolProj2.GetSolution(fieldType=PFN.displacement)

    relaError=np.linalg.norm(sol1-projSol)/np.linalg.norm(sol1)
    npt.assert_array_less(np.abs(relaError), 1e-5, "Relative error too large")
    l2Error=BeamSolProj2.ComputeSpatialErrorIndicator(sol1-projSol,"L2")
    npt.assert_array_less(np.abs(l2Error), 1e-6, "L2 error too large")
    return "ok"

def CheckIntegrity_BeamNodalForcesAuxiliaryField():
    BeamProblem=GetfemMecaProblem()
    BeamProblem.mesh,BeamProblem.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    BeamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    BeamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    BeamProblem.SetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces,params={"boundary":BeamProblem.refNumByRegion["LEFT"]})
    BeamProblem.Preprocessing(variables=femVariables)
    BeamProblem.BuildModel()
    BeamProblem.RunProblem()
    sol0=BeamProblem.GetSolution(fieldType=PFN.displacement)
    nodalForces=BeamProblem.GetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces)
    BeamProblem.ExportSolutionInFile(filename="toto.csv")

    BeamProblem2=type(BeamProblem)(BeamProblem)
    assert BeamProblem.auxiliaryFieldGeneration==BeamProblem2.auxiliaryFieldGeneration
    BeamProblem2.BuildModel()
    BeamProblem2.RunProblem()
    BeamProblem2.ExportSolutionInFile(filename="tata.csv")
    return "ok"

def CheckIntegrity_BeamNodalForcesAuxiliaryFieldDegree2():
    BeamProblem=GetfemMecaProblem()
    BeamProblem.mesh,BeamProblem.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    BeamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    BeamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    BeamProblem.SetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces,params={"boundary":BeamProblem.refNumByRegion["LEFT"]})
    BeamProblem.Preprocessing()
    BeamProblem.BuildModel()
    BeamProblem.RunProblem()
    sol0=BeamProblem.GetSolution(fieldType=PFN.displacement)
    nodalForces=BeamProblem.GetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces)
    BeamProblem.ExportSolutionInFile(filename="toto.csv")
    return "ok"

def CheckIntegrity():

    totest = [
    CheckIntegrity_solutionContactComputed,
    CheckIntegrity_solutionLinearElasticity,
    CheckIntegrity_solutionContactComputed,
    CheckIntegrity_solutionContactInclinedFoundation,
    CheckIntegrity_VariableForcesolutionComputed,
    CheckIntegrity_NeumannForceSolutionComputed,
    CheckIntegrity_RollingContactLinearAnglePiloted,
    CheckIntegrity_RollingContactDIS_Rolling,
    CheckIntegrity_RollingContactFORC_Rolling,
    CheckIntegrity_InitWithSolutionLinearElast,
    CheckIntegrity_InitWithSolutionContact,
    CheckIntegrity_InitWithSolutionContactNoMultiplier,
    CheckIntegrity_BeamNodalForcesFromDirichlet,
    CheckIntegrity_BeamNodalForcesFromNeumann,
    CheckIntegrity_BeamSolProjection,
    CheckIntegrity_BeamNodalForcesAuxiliaryField,
    CheckIntegrity_BeamNodalForcesAuxiliaryFieldDegree2
              ]

    for test in totest:
        res =  test()
        if  res.lower() != "ok" :
            return res

    return "OK"

if __name__ == '__main__':
    print(CheckIntegrity())