#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This file introduce a Getfem++ interface.
#The idea is to combine the individual features relying on Getfem++ "bricks" to build a physical problem

import numpy as np
import math
import os
from copy import deepcopy

from lips.physical_simulator.GetfemSimulator.GetfemProblemBase import GetfemProblemBase
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

from lips.physical_simulator.GetfemSimulator.GetfemBricks.Utilities import ComputeDirichletNodalForces
import lips.physical_simulator.GetfemSimulator.GetfemBricks.FeSpaces as gfFeSpace
import lips.physical_simulator.GetfemSimulator.GetfemBricks.BehaviourLaw as gfBehaviour
import lips.physical_simulator.GetfemSimulator.GetfemBricks.ModelTools as gfModel
import lips.physical_simulator.GetfemSimulator.GetfemBricks.ExportTools as gfExport
import lips.physical_simulator.GetfemSimulator.GetfemBricks.ExternalConditions as gfExternal
import lips.physical_simulator.GetfemSimulator.GetfemBricks.RollingConditions as gfRolling

from lips.physical_simulator.GetfemSimulator.Utilitaries import ComputeNorm2

class ParamNotFound(Exception):
    pass

class GetfemMecaProblem(GetfemProblemBase):
    def __init__(self,name=None,auxiliaryOutputs=None,other=None):
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

        super(GetfemMecaProblem,self).__init__(name,auxiliaryOutputs,other)
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
            self.incompressibility = False
        else:
            self._materials = deepcopy(other._materials)
            self._dirichlet = deepcopy(other._dirichlet)
            self._neumann = deepcopy(other._neumann)
            self._contact = deepcopy(other._contact)
            self._sources = deepcopy(other._sources)
            self.problemCharacsByType = deepcopy(other.problemCharacsByType)
            self.nodalForces= deepcopy(other.nodalForces)
            self.incompressibility = other.incompressibility

    def GetFieldFromName(self,fieldName):
        if fieldName==PFN.equivDirichletNodalForces:
            return ComputeDirichletNodalForces(self,self.auxiliaryParams[fieldName]["boundary"])
        elif fieldName==PFN.contactUnilateralMultiplier or fieldName==PFN.contactMultiplier:
            return self.GetSolution(fieldName)
        elif fieldName==PFN.displacementNorm:
            displacement=self.GetSolution(PFN.displacement)
            return ComputeNorm2(displacement)
        else:
            raise Exception("Field "+fieldName+" is not available")

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
        if PFN.contactMultiplier in self.solutions.keys():
            multiplierType=PFN.contactMultiplier
        elif PFN.contactUnilateralMultiplier in self.solutions.keys():
            multiplierType=PFN.contactUnilateralMultiplier
        else:
            raise Exception("Type of multiplier not covered yet")

        u,mult = self.GetSolution(PFN.displacement),self.GetSolution(multiplierType)
        mfu,mfmult = self.GetFeSpace(PFN.displacement),self.GetFeSpace(multiplierType)
        contactBoundName=self.contact[0][0]
        refIndices=self.GetFeSpace(multiplierType).basic_dof_on_region(self.refNumByRegion[contactBoundName])
        extendedLambda=np.zeros(gfFeSpace.GetNbDof(mfmult))
        extendedLambda[refIndices]=-mult
        gfExport.ExportFieldInGmsh(filename=filename,mfu=mfu,U=u,mfField=mfmult,field=extendedLambda,fieldName="Lagrange multiplier")

    def ContactMultipliersRequired(self,contactType):        
        return gfExternal.MultiplierRequiredConditionLookUp.MultiplierVariableRequired(contactType)

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
        self.multiplierVariable=multiplierVariable

        if variables is not None:
            for varName,val in variables.items():
                self.feDef[varName]=val

    def DefineModel(self):
        model=super(GetfemMecaProblem,self).DefineModel()
        gfBehaviour.AddFiniteStrainMacroToModel(model)
        return model

    def BuildModel(self):
        """
        .. py:method:: BuildModel()

        Clean model and build constitutive brick one by one, sequentially
        """
        self.CleanProblemModel()
        self.BuildBricks()

    def BuildBricks(self):
        self.materialsbricks=self.BuildBehaviourLaw()
        self.bodyForcesBrick=self.BuildBodyForces()
        self.dirichletBrick=self.BuildDirichletBC()
        self.neumannBrick=self.BuildNeumannBC()
        self.contactBrick=self.BuildContactBC()
        if self.nodalForces:
            self.nodalForcesBrick=self.BuildNodalForces()
        if self.incompressibility:
            self.incompressibilityBrick=self.AddIncompressibilityCondition()

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
        gfModel.DeleteBoundaryCondition(self.model,brickTodelete)

    def BuildBehaviourLaw(self):
        """
        .. py:method:: BuildBehaviourLaw()

        Build behaviour law bricks
        """
        self.problemCharacsByType["Material"]=self.materials
        problemParamsForBuild={"model":self.model,
                       "integrationMethod":self.integrMethods["standard"],
                       "dimension":self.dim,
                       "tagMap":self.refNumByRegion}

        materialsbricks=[]
        for materialId,(tagname,material) in enumerate(self.materials):
            materialLaw=material["law"]
            materialParams={k: material[k] for k in set(list(material.keys())) - set(["law"])}
            materialParams["regionTag"]=(tagname,materialId)
            materialbrick=gfBehaviour.CreateBehaviourLawBrick(materialLaw)
            materialbrickId=materialbrick(problemParams=problemParamsForBuild,brickParams=materialParams)
            materialsbricks.append(materialbrickId)
        return materialsbricks


    def BuildBodyForces(self):
        """
        .. py:method:: BuildBodyForces()

        Build body forces bricks
        """
        self.problemCharacsByType["Sources"]=self.sources
        problemParamsForBuild={"model":self.model,
                       "integrationMethod":self.integrMethods["standard"],
                       "dimension":self.dim,
                       "tagMap":self.refNumByRegion}

        bodyForcesBricks=[]
        for sourceId,(tagname,source) in enumerate(self.sources):
            sourceType=source["type"]
            sourceParams={k: source[k] for k in set(list(source.keys())) - set(["type"])}
            sourceParams["regionTag"]=(tagname,sourceId)
            bodyForcesBrick=gfExternal.CreateSourceBrick(sourceType)
            bodyForcesBrickId=bodyForcesBrick(problemParams=problemParamsForBuild,brickParams=sourceParams)
            bodyForcesBricks.append(bodyForcesBrickId)
        return bodyForcesBricks

    def BuildDirichletBC(self):
        """
        .. py:method:: BuildDirichletBC()

        Build dirichlet bricks
        """
        self.problemCharacsByType["Dirichlet"]=self.dirichlet
        problemParamsForBuild={"model":self.model,
                           "integrationMethod":self.integrMethods["standard"],
                           "feSpace":self.GetFeSpace(PFN.displacement),
                           "dimension":self.dim,
                           "tagMap":self.refNumByRegion
                        }

        dirichletBricks=[]
        for dirichId,(tagname,dirich) in enumerate(self.dirichlet):
            dirichType=dirich["type"]
            dirichParams={k: dirich[k] for k in set(list(dirich.keys())) - set(["type"])}
            dirichParams["regionTag"]=(tagname,dirichId)
            dirichletBrick=gfExternal.CreateDirichletBrick(dirichType)
            dirichletBrickId=dirichletBrick(problemParams=problemParamsForBuild, brickParams=dirichParams)
            dirichletBricks.append(dirichletBrickId)
        return dirichletBricks

    def BuildNeumannBC(self):
        """
        .. py:method:: BuildNeumannBC()

        Build neumann bricks
        """
        self.problemCharacsByType["Neumann"]=self.neumann
        problemParamsForBuild={"model":self.model,
                       "integrationMethod":self.integrMethods["standard"],
                       "feSpace":self.GetFeSpace("rim"),
                       "dimension":self.dim,
                       "tagMap":self.refNumByRegion
                           }

        neumannBricks=[]
        for neumannId,(tagname,neum) in enumerate(self.neumann):
            neumType=neum["type"]
            neumParams={k: neum[k] for k in set(list(neum.keys())) - set(["type"])}
            neumParams["regionTag"]=(tagname,neumannId)
            neumannBrick=gfExternal.CreateNeumannBrick(neumType)
            neumannBrickId=neumannBrick(problemParams=problemParamsForBuild, brickParams=neumParams)
            neumannBricks.append(neumannBrickId)
        return neumannBricks

    def BuildNodalForces(self):
        """
        .. py:method:: BuildNodalForces()

        Build nodal forces bricks
        """
        nodalForcesBricks=[]
        for _,nodalForce in self.nodalForces:
            nodalForcesBrick=gfModel.AddExplicitRHS(self.model,'u',nodalForce)
            nodalForcesBricks.append(nodalForcesBrick)
        return nodalForcesBricks

    def BuildContactBC(self):
        """
        .. py:method:: BuildContactBC()

        Build contact bricks
        """
        self.problemCharacsByType["Contact"]=self.contact
        problemParamsForBuild={"model":self.model,
                       "integrationMethod":self.integrMethods["composite"],
                       "dimension":self.dim,
                       "tagMap":self.refNumByRegion
                           }

        contactBricks=[]
        for contId,(tagname,cont) in enumerate(self.contact):
            contactType=cont["type"]
            self.DefineContactFilteredSpace(contactType=contactType,contactTagname=tagname)
            problemParamsForBuild["feSpace"]=self.GetFeSpace("obstacle")
            contParams={k: cont[k] for k in set(list(cont.keys())) - set(["type"])}
            contParams["regionTag"]=(tagname,contId)
            contactBrick=gfExternal.CreateContactBrick(contactType)
            contactBrickId=contactBrick(problemParams=problemParamsForBuild, brickParams=contParams)
            contactBricks.append(contactBrickId)
        return contactBricks

    def DefineContactFilteredSpace(self,contactType,contactTagname):
        multVar=self.ContactMultipliersRequired(contactType)
        if multVar is not None:
            if self.multiplierVariable is not None:
                multVariable=self.multiplierVariable
            else:
                multVariable=("lambda",multVar,contactTagname) 
            self.spacesVariables.append(multVariable) 

            if multVariable[1]==PFN.contactMultiplier:
                dof=self.dim
            elif multVariable[1]==PFN.contactUnilateralMultiplier:
                dof=1
            else:
                raise Exception("Case not covered")
            self.feDef[multVariable[1]]={"degree": 1, "dof":dof}
            space=self.feDef[multVariable[1]]
            
            self.feSpaces[multVariable[1]]=gfFeSpace.DefineFESpaces(mesh=self.mesh,elements_degree=space["degree"],dof=space["dof"])

            existingVariables=[variable[0] for variable in self.variables]
            if "lambda" not in existingVariables:
                contactBound=self.refNumByRegion[multVariable[-1]]
                variable=(multVariable[0],multVariable[1],contactBound)
                self.AddVariable(variable)

    def AddIncompressibilityCondition(self):
        self.problemCharacsByType["Incompressibility"]=[["ALL",{"finite strain incompressibility":True}]]        
        
        mfp=gfFeSpace.DefineClassicalDiscontinuousFESpace(mesh=self.mesh,elements_degree=1,dof=1)
        gfModel.AddVariable(model=self.model,var="p",mfvar=mfp)
        incompressibilityBrick=gfBehaviour.AddIncompressibilityBrick(model=self.model,mim=self.integrMethods["standard"])
        return incompressibilityBrick

    def GetAllPhyComponents(self):
        """
        .. py:method:: GetAllComponents()

        Get all existing physical component of the problem
        """
        return [self.materials,self.sources,self.dirichlet,self.neumann,self.contact]

    def GetProblemState(self,paramName):
        """
        .. py:method:: GetProblemState()

        Get the value of an existing physical parameter
        :param string paramName: name of the parameter        
        """
        allComponents=self.GetAllPhyComponents()
        value=[params[paramName] for component in allComponents for tag,params in component  if paramName in params]
        if len(value)==1:
            return value[0]
        elif not len(value):
            return self.GetAmbiguousParameter(paramName,allComponents)
        else:
            raise Exception("Ambiguous: several values available for "+paramName+" parameter, not implemented yet")

    def GetAmbiguousParameter(self,paramName,components):
        paramNameAmbiguous,paramPositionInComponent=self.ExtractNameAndPositionInComponent(paramName=paramName)
        foundParams=0
        for component in components:
            if len(component)<paramPositionInComponent+1 or not component:
                continue
            targetComponent=component[paramPositionInComponent]
            _,targetParams=targetComponent
            if paramNameAmbiguous in targetParams:
                value=targetParams[paramNameAmbiguous]
                foundParams+=1
        if not foundParams:
            raise ParamNotFound("parameter "+paramName+" was really not found")
        elif foundParams>1:
            raise Exception("Ambiguous: several values available for "+paramName+" parameter, not implemented yet")
        else:
            print("Parameter "+paramName+" was found")
            return value

    def SetProblemState(self,valueByParams):
        """
        .. py:method:: SetProblemState()

        Set the value of existing physical parameters
        :param dict valueByParams: parameter value with respect to parameter name        
        """
        allComponents=self.GetAllPhyComponents()
        for paramName,paramVal in valueByParams.items():
            foundParams=0
            for component in allComponents:
                for _,params in component:
                    if paramName in params:
                        params[paramName]=paramVal
                        foundParams+=1

            if not foundParams:
                print("Warning: parameter "+paramName+" not found. Assume ambiguity in name and try again")
                self.SetAmbiguousParameter(paramName,paramVal,allComponents)
            elif foundParams>1:
                raise Exception("Ambiguous: "+str(foundParams)+" occurences found for "+paramName+" parameter, not implemented yet")
            else:
                pass

    def SetAmbiguousParameter(self,paramName,paramVal,components):
        paramNameAmbiguous,paramPositionInComponent=self.ExtractNameAndPositionInComponent(paramName=paramName)
        foundParams=0
        for component in components:
            if len(component)<paramPositionInComponent+1 or not component:
                continue
            targetComponent=component[paramPositionInComponent]
            _,targetParams=targetComponent
            if paramNameAmbiguous in targetParams:
                targetParams[paramNameAmbiguous]=paramVal
                foundParams+=1
        if not foundParams:
            raise ParamNotFound("parameter "+paramName+" was really not found")
        elif foundParams>1:
            raise Exception("Ambiguous: "+str(foundParams)+" occurences found for "+paramName+" parameter, not implemented yet")
        else:
            print("Parameter "+paramName+" was found")

    def ExtractNameAndPositionInComponent(self,paramName):
        splittedName=paramName.split("_")
        if len(splittedName)==1:
            raise ParamNotFound("parameter "+paramName+" was really not found")
        else:
            splittedName=["_".join(splittedName[0:len(splittedName)-1]),splittedName[-1]]
        paramNameAmbiguous,paramPosition=splittedName
        paramPositionInComponent=int(paramPosition)
        return paramNameAmbiguous,paramPositionInComponent


    def SaveProblemConfig(self,filename):
        problemConfig=self.GetProblemConfig()
        np.save(filename, problemConfig)

    def LoadProblemConfig(self,filename):
        problemConfig = np.load(filename,allow_pickle='TRUE').item()
        self.SetProblemConfig(problemConfig=problemConfig)

    def GetProblemConfig(self):
        problemConfig={
            "materials":self.materials,
            "dirichlet":self.dirichlet,
            "neumann":self.neumann,
            "contact":self.contact,
            "sources":self.sources,
            "incompressibility":self.incompressibility,
            "nodalForces":self.nodalForces
        }
        return problemConfig

    def SetProblemConfig(self,problemConfig):
        self.materials=problemConfig["materials"]
        self.dirichlet=problemConfig["dirichlet"]
        self.neumann=problemConfig["neumann"]
        self.contact=problemConfig["contact"]
        self.sources=problemConfig["sources"]
        self.incompressibility=problemConfig["incompressibility"]
        self.nodalForces=problemConfig["nodalForces"]

    def WriteSolutionInFile(self,filename):
        """
        .. py:method:: WriteSolutionInFile(filename)

        Export in .csv format a field computed using Getfem++
        :param string filename: output file name         
        """
        self.ExportDisplacementInFile(filename)
        self.ExportAuxiliaryFieldsInFile(filename)

    def ExportDisplacementInFile(self,filename):
        fieldType=PFN.displacement
        filePath=filename.split(os.sep)
        filePath[-1]=fieldType+"_"+filePath[-1]+".csv"
        rootDir,path=filePath[0],filePath[1:]
        newFileName=os.path.join(os.sep, rootDir+os.sep, *path)
        feSpace,solution=self.GetFeSpace(fieldType),self.GetSolution(fieldType)
        gfExport.ExportSolutionInCSV(newFileName,"VectorSolution",feSpace,solution)

    def ExportAuxiliaryFieldsInFile(self,filename):
        for fieldName,tag in self.auxiliaryFieldGeneration.items():
            if tag:
                filePath=filename.split(os.sep)
                filePath[-1]=fieldName+"_"+filePath[-1]+".csv"
                rootDir,path=filePath[0],filePath[1:]
                newFileName=os.path.join(os.sep, rootDir+os.sep, *path)
                if fieldName==PFN.contactUnilateralMultiplier or fieldName==PFN.contactMultiplier:
                    if len(self.contact)>1:
                        raise Exception("Several contact label case not handled yet!")
                    tagname,_ =self.contact[0]
                    feSpace,solution=self.GetFeSpace(fieldName),self.GetAuxiliaryField(fieldName=fieldName)
                    if fieldName==PFN.contactUnilateralMultiplier:
                        gfExport.ExportUnilateralMultipliersInCSV(newFileName,feSpace,solution,self.refNumByRegion[tagname])
                    elif fieldName==PFN.contactMultiplier:
                        gfExport.ExportNDofMultipliersInCSV(newFileName,feSpace,solution,self.refNumByRegion[tagname])
                elif fieldName==PFN.displacementNorm:
                    feSpace,solution=self.GetFeSpace(PFN.displacement),self.GetAuxiliaryField(fieldName=fieldName)
                    gfExport.ExportSolutionInCSV(newFileName,'ScalarField',feSpace,solution)
                elif fieldName==PFN.equivDirichletNodalForces:
                    feSpace,fieldVal=self.GetFeSpace(PFN.displacement),self.GetAuxiliaryField(fieldName=fieldName)
                    gfExport.ExportNodalStressFieldInCSV(newFileName,"VectorSolution",feSpace,fieldVal)
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
    def __init__(self,name=None,auxiliaryOutputs=None,other=None):
        """
        .. py:method:: __init__(other)
        :param GetfemRollingWheelProblem other: another instance of this very class        

        Constructor of the class GetfemRollingWheelProblem
        :ivar list rolling: rolling conditions
        """
        super(GetfemRollingWheelProblem,self).__init__(name=name,auxiliaryOutputs=auxiliaryOutputs,other=other)

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

        tagname,roll=self.rolling
        rollingType=roll["type"]
        rollingParams={k: roll[k] for k in set(list(roll.keys())) - set(["type"])}
        rollingParams["regionTag"]=(tagname,None)
        problemParams={"model":self.model,
                           "integrationMethod":self.integrMethods["standard"],
                           "feSpace":self.GetFeSpace(PFN.displacement),
                           "dimension":self.dim,
                           "tagMap":self.refNumByRegion
                           }

        rollingBrick=gfRolling.CreateRollingBrick(rollingType)
        rollingBrickIds=rollingBrick(problemParams=problemParams,brickParams=rollingParams)
        return rollingBrickIds

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

    def BuildBricks(self):
        """
        .. py:method:: BuildBricks()

        Build constitutive brick one by one, sequentially
        """
        super().BuildBricks()
        self.HandleRolling()

    def HandleRolling(self):
        rollingType=self.GetRollingConditionType()
        if rollingType=="AnglePiloted":
            pass
        elif rollingType=="DIS_Rolling" or rollingType=="FORC_Rolling":
            self.rollingBrick=self.BuildRollingBC()
        elif rollingType=="TwoSteps_FORC_Rolling":
            #First block: loading
            problemParamsForBuild={"model":self.model,
                       "integrationMethod":self.integrMethods["standard"],
                       "feSpace":self.GetFeSpace("rim"),
                       "dimension":self.dim,
                       "tagMap":self.refNumByRegion
                           }
            tagname,neumParams=self.rolling
            neumParams={k: neumParams[k] for k in set(list(neumParams.keys())) - set(["type"])}
            neumParams["regionTag"]=(tagname,1000)
            neumannBrick=gfExternal.CreateNeumannBrick("RimRigidityNeumann")
            self.loadingBrick=neumannBrick(problemParams=problemParamsForBuild, brickParams=neumParams)
            #Second block: rolling
            self.rolling[1]['type'] = "FORC_Rolling_Penalized"
            self.rollingBrick=self.BuildRollingBC()
        else:
            raise Exception("Do not know how to handle "+str(rollingType)+" rolling condition")

    def RunProblem(self,options=None):
        """
        .. py:method:: RunProblem()

        Solve physical problem
        """
        rollingType=self.GetRollingConditionType()
        if rollingType=="AnglePiloted":
            solverState=self.StandardSolve(options=options)
            self.DeleteModelBrickFromId(self.dirichletBrick)
            self.rollingBrick=self.BuildRollingBC()
            solverState=self.StandardSolve(options=options)
        elif rollingType=="DIS_Rolling" or rollingType=="FORC_Rolling_Penalized":
            solverState=self.StandardSolve(options=options)
        elif rollingType=="FORC_Rolling":
            solverState=gfModel.TwoStepsRollingSolve(self)
        else:
            raise Exception("Do not know how to handle "+str(rollingType)+" rolling condition")

        self.SaveSolutions()
        self.SaveAuxiliaryFields()
        return solverState

    def GetProblemConfig(self):
        problemConfig={
            "materials":self.materials,
            "dirichlet":self.dirichlet,
            "neumann":self.neumann,
            "contact":self.contact,
            "sources":self.sources,
            "incompressibility":self.incompressibility,
            "nodalForces":self.nodalForces,
            "rolling":self.rolling
        }
        return problemConfig

    def SetProblemConfig(self,problemConfig):
        self.materials=problemConfig["materials"]
        self.dirichlet=problemConfig["dirichlet"]
        self.neumann=problemConfig["neumann"]
        self.contact=problemConfig["contact"]
        self.sources=problemConfig["sources"]
        self.incompressibility=problemConfig["incompressibility"]
        self.nodalForces=problemConfig["nodalForces"]
        self.rolling=problemConfig["rolling"]

#################################Test#################################
import numpy.testing as npt

import lips.physical_simulator.GetfemSimulator.GetfemBricks.MeshTools as gfMesh
from lips.physical_simulator.GetfemSimulator.GetfemBricks.Utilities import ProjectSolOnMesh

def CheckIntegrity_ProblemState():
    print("Testing problem state")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ],["ImaginaryMaterial2", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}],["ImaginaryZone2",{"type" : "Uniform","source_x":100.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]

    youngVal=WPM.GetProblemState("young_0")
    npt.assert_equal(youngVal,21E6)
    poissonVal=WPM.GetProblemState("poisson_1")
    npt.assert_equal(poissonVal,0.3)
    dispVal=WPM.GetProblemState("Disp_Amplitude")
    npt.assert_equal(dispVal,6)

    npt.assert_raises(ParamNotFound, WPM.GetProblemState, "toto")

    WPM.SetProblemState({"young_0":10E6,"poisson_1":0.25})
    youngVal=WPM.GetProblemState("young_0")
    npt.assert_equal(youngVal,10E6)
    poissonVal=WPM.GetProblemState("poisson_1")
    npt.assert_equal(poissonVal,0.25)

    WPMSetter=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPMSetter.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPMSetter.refNumByRegion=refNumByRegion
    WPMSetter.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ],["ImaginaryMaterial2", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPMSetter.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}],["ImaginaryZone2",{"type" : "Uniform","source_x":100.0,"source_y":0.0}] ]
    WPMSetter.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPMSetter.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]

    npt.assert_raises(ParamNotFound, WPM.SetProblemState, {"toto":3})
    WPMSetter.SetProblemState({"young_0":3e10,"poisson_1":0.4,"source_x_1":200.0})
    assert WPMSetter.materials==[["ALL", {"law":"LinearElasticity","young":3e10,"poisson":0.3} ],["ImaginaryMaterial2", {"law":"LinearElasticity","young":21E6,"poisson":0.4} ]]
    assert WPMSetter.sources==[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}],["ImaginaryZone2",{"type" : "Uniform","source_x":200.0,"source_y":0.0}] ]

    return "ok"

def CheckIntegrity_solutionLinearElasticity():
    print("Testing linear elasticity case")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
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
    WPM.SetAuxiliaryField(fieldName=PFN.contactMultiplier)
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
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
    currentPath=os.path.abspath(os.getcwd())
    WPM.WriteSolutionInFile(currentPath+os.sep+"solution")
    return "ok"

def CheckIntegrity_solutionContactInclinedFoundation():
    print("Testing linear elasticity with unilateral contact and friction case with oriented foundation")
    WPM=GetfemMecaProblem()
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
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
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
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
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions, meshSize=1, RefNumByRegion=refNumByRegion)
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
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
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
    WPMCopy=type(WPM)(other=WPM)
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
    WPM.mesh=gfMesh.GenerateWheelMeshRolling(wheelDimensions=wheelDimensions,meshSize=2,RefNumByRegion=refNumByRegion)
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
    WPM.mesh=gfMesh.GenerateWheelMeshRolling(wheelDimensions=wheelDimensions,meshSize=2,RefNumByRegion=refNumByRegion)
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
    # mim=WPM.integrMethods["composite"]
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
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    print("Solver state: ",state)

    disp = WPM.GetSolution(PFN.displacement)

    WPMCopy=type(WPM)(other=WPM)
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
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
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

    WPMCopy=type(WPM)(other=WPM)
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
    WPM.mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":1.,"poisson":0.3} ]]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "PlanePenalized","fricCoeff":0.4}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    WPM.RunProblem()
    disp = WPM.GetSolution(PFN.displacement)

    WPMCopy=type(WPM)(other=WPM)
    WPMCopy.BuildModel()
    WPMCopy.InitVariable(PFN.displacement,disp)
    WPMCopy.RunProblem()
    disp2 =WPMCopy.GetSolution(PFN.displacement)
    errorDisp=np.nan_to_num(np.abs(disp2 - disp) / np.abs(disp) )

    npt.assert_array_less(np.percentile(errorDisp,99),1e-2)
    return "ok"

def CheckIntegrity_BeamNodalForcesFromDirichlet():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=gfMesh.GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}],["RIGHT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    beamProblem.Preprocessing(variables=femVariables)
    beamProblem.SetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces,params={"boundary":beamProblem.refNumByRegion["LEFT"]})
    beamProblem.BuildModel()
    beamProblem.RunProblem()
    sol0=beamProblem.GetSolution(fieldType=PFN.displacement)
    nodalForces=beamProblem.GetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces)

    beamProblem2=GetfemMecaProblem()
    beamProblem2.mesh,beamProblem2.refNumByRegion=beamProblem.mesh,beamProblem.refNumByRegion
    beamProblem2.materials=beamProblem.materials
    beamProblem2.dirichlet=[["RIGHT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    beamProblem2.neumann=beamProblem.neumann
    beamProblem2.nodalForces=[["ALL",nodalForces]]
    beamProblem2.Preprocessing(variables=femVariables)
    beamProblem2.BuildModel()
    beamProblem2.RunProblem()
    sol1=beamProblem2.GetSolution(fieldType=PFN.displacement)
    relaError=np.linalg.norm(sol1-sol0)/np.linalg.norm(sol0)
    absError=np.linalg.norm(sol1-sol0)
    npt.assert_array_less(np.abs(relaError), 1e-6, "Relative error too large")
    npt.assert_array_less(np.abs(absError), 1e-6, "Absolute error too large")
    return "ok"

def CheckIntegrity_BeamNodalForcesFromNeumann():
    BeamNeumannProblem=GetfemMecaProblem()
    BeamNeumannProblem.mesh,BeamNeumannProblem.refNumByRegion=gfMesh.GenerateSimpleMesh(meshSize=20.0)
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

def CheckIntegrity_beamSolProjection():
    beamSolProj=GetfemMecaProblem()
    coarseMesh,numByRegion=gfMesh.GenerateSimpleMesh(meshSize=32.0)
    beamSolProj.mesh,beamSolProj.refNumByRegion=coarseMesh,numByRegion
    beamSolProj.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    beamSolProj.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}]]
    beamSolProj.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    beamSolProj.Preprocessing()
    beamSolProj.BuildModel()
    beamSolProj.RunProblem()
    sol0=beamSolProj.GetSolution(fieldType=PFN.displacement)

    refinedMesh,numByRegion=gfMesh.GenerateSimpleMesh(meshSize=64.0)
    projSol=ProjectSolOnMesh(coarseSol=sol0,solDegree=2,coarseMesh=coarseMesh,refMesh=refinedMesh)

    beamSolProj2=GetfemMecaProblem()
    beamSolProj2.mesh,beamSolProj2.refNumByRegion=refinedMesh,numByRegion
    beamSolProj2.materials=beamSolProj.materials
    beamSolProj2.dirichlet=beamSolProj.dirichlet
    beamSolProj2.neumann=beamSolProj.neumann
    beamSolProj2.Preprocessing()
    beamSolProj2.BuildModel()
    beamSolProj2.RunProblem()
    sol1=beamSolProj2.GetSolution(fieldType=PFN.displacement)

    relaError=np.linalg.norm(sol1-projSol)/np.linalg.norm(sol1)
    npt.assert_array_less(np.abs(relaError), 1e-5, "Relative error too large")
    l2Error=beamSolProj2.ComputeSpatialErrorIndicator(sol1-projSol,"L2")
    npt.assert_array_less(np.abs(l2Error), 1e-6, "L2 error too large")
    return "ok"

def CheckIntegrity_BeamNodalForcesAuxiliaryField():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=gfMesh.GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    beamProblem.SetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces,params={"boundary":beamProblem.refNumByRegion["LEFT"]})
    beamProblem.Preprocessing(variables=femVariables)
    beamProblem.BuildModel()
    beamProblem.RunProblem()
    sol0=beamProblem.GetSolution(fieldType=PFN.displacement)
    nodalForces=beamProblem.GetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces)
    fullPath=os.path.join(os.path.abspath("."), "toto")
    beamProblem.WriteSolutionInFile(filename=fullPath)

    beamProblem2=type(beamProblem)(other=beamProblem)
    assert beamProblem.auxiliaryFieldGeneration==beamProblem2.auxiliaryFieldGeneration
    beamProblem2.BuildModel()
    beamProblem2.RunProblem()
    fullPath=os.path.join(os.path.abspath("."), "tata")
    beamProblem2.WriteSolutionInFile(filename=fullPath)
    return "ok"

def CheckIntegrity_BeamNodalForcesAuxiliaryFieldDegree2():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=gfMesh.GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    beamProblem.SetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces,params={"boundary":beamProblem.refNumByRegion["LEFT"]})
    beamProblem.Preprocessing()
    beamProblem.BuildModel()
    beamProblem.RunProblem()
    sol0=beamProblem.GetSolution(fieldType=PFN.displacement)
    nodalForces=beamProblem.GetAuxiliaryField(fieldName=PFN.equivDirichletNodalForces)
    fullPath=os.path.join(os.path.abspath("."), "toto")
    beamProblem.WriteSolutionInFile(filename=fullPath)
    return "ok"

def CheckIntegrity_CompressibleNeoHookean():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=gfMesh.GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"CompressibleNeoHookean","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_Amplitude":0.0, "Disp_Angle":0.0}] ]
    beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    beamProblem.Preprocessing(variables=femVariables)
    beamProblem.BuildModel()
    beamProblem.RunProblem()
    return "ok"

def CheckIntegrity_SaveLoadConfig():
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    mesh=gfMesh.GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)

    WPM=GetfemMecaProblem()
    WPM.mesh=mesh
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]

    myTrueConfig=WPM.GetProblemConfig()
    WPM.SaveProblemConfig(filename="contactproblem.npy")
    WPMBis=type(WPM)()
    WPMBis.mesh=mesh
    WPMBis.LoadProblemConfig(filename="contactproblem.npy")
    myLoadedConfig=WPMBis.GetProblemConfig()
    assert(myTrueConfig==myLoadedConfig)

    wheelDimensions=(8., 15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    mesh=gfMesh.GenerateWheelMeshRolling(wheelDimensions=wheelDimensions,meshSize=2,RefNumByRegion=refNumByRegion)
    WPMR=GetfemRollingWheelProblem()
    WPMR.mesh=mesh
    WPMR.refNumByRegion=refNumByRegion
    WPMR.materials = [["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1}]]
    WPMR.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPMR.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.6}] ]
    WPMR.rolling=["HOLE_BOUND",{"type" : "DIS_Rolling", "theta_Rolling":150., 'd': 1.,'currentTime':0.0}]

    myTrueConfig=WPMR.GetProblemConfig()
    WPMR.SaveProblemConfig(filename="rollingContactproblem.npy")
    WPMRBis=type(WPMR)()
    WPMRBis.mesh=mesh
    WPMRBis.LoadProblemConfig(filename="rollingContactproblem.npy")
    myLoadedConfig=WPMRBis.GetProblemConfig()
    assert(myTrueConfig==myLoadedConfig)
    return "ok"

def CheckIntegrity_Run3DCase():
    WPM=GetfemMecaProblem()
    WPM.mesh,WPM.refNumByRegion=gfMesh.GenerateTagged3DBeam(meshSize=4)
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.dirichlet=[["XMIN",{"type" : "GlobalVector", "val_x":0, "val_y":0, "val_z":0}] ]
    WPM.neumann=[["ZMAX",{"type" : "StandardNeumann", "fx":0.0, "fy":0.0, "fz":0.0}] ]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":-1e6,"source_z":0.0}] ]
    WPM.contact=[ ["YMIN",{"type" : "Plane","gap":0.0,"fricCoeff":0.0}] ]

    WPM.Preprocessing()
    WPM.BuildModel()
    WPM.RunProblem()
    WPM.ExportSolutionWithMultipliersInGmsh(filename="3DBeamSolution.pos")
    return "ok"

def CheckIntegrity_Wheel3D():
    WPM=GetfemMecaProblem()

    from lips.physical_simulator.GetfemSimulator.MeshGenerationTools import Standard3DWheelGenerator
    mesh3DConfig={"wheel_Dimensions":(8.,15.,4),
                "mesh_size":1.0}
    gmshMesh=Standard3DWheelGenerator(**mesh3DConfig)
    gmshMesh.GenerateMesh(outputFile="my3DWheel")

    mesh=gfMesh.ImportGmshMesh(meshFile="my3DWheel.msh")
    refNumByRegion=gmshMesh.tagMap
    # from lips.physical_simulator.GetfemSimulator.GetfemBricks.PlotTools import PlotTagsInMesh
    # PlotTagsInMesh(mesh=mesh,refNumByRegion=refNumByRegion,tagsToPlot="Exterior",plotPoints=False)
    WPM.mesh=mesh
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":-20e4,"source_z":0.0}] ]
    WPM.dirichlet=[["Interior",{"type" : "scalar", "Amplitude":1, "ThetaAngle":math.pi/2, "PhiAngle":-math.pi/2}] ]
    #WPM.contact=[ ["Exterior",{"type" : "Plane","gap":0.0,"fricCoeff":0.0}] ]
    WPM.contact=[ ["Exterior",{"type" : "PlanePenalized","fricCoeff":0.9,"PenalizationParam":10e7}] ]

    WPM.Preprocessing()
    WPM.BuildModel()
    WPM.RunProblem()

    WPM.ExportSolution(filename="3DWheel.pos", extension="gmsh")

    return "ok"

def CheckIntegrity():

    totest = [
    CheckIntegrity_ProblemState,
    CheckIntegrity_solutionContactComputed,
    CheckIntegrity_solutionLinearElasticity,
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
    CheckIntegrity_beamSolProjection,
    CheckIntegrity_BeamNodalForcesAuxiliaryField,
    CheckIntegrity_BeamNodalForcesAuxiliaryFieldDegree2,
    CheckIntegrity_CompressibleNeoHookean,
    CheckIntegrity_SaveLoadConfig,
    CheckIntegrity_Run3DCase,
    CheckIntegrity_Wheel3D
              ]

    for test in totest:
        res =  test()
        if  res.lower() != "ok" :
            return res

    return "OK"

if __name__ == '__main__':
    print(CheckIntegrity())