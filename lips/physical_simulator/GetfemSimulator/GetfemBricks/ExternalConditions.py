#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.spatial import cKDTree

from BasicTools.Helpers.Factory import Factory

import getfem as gf
gf.util_trace_level(1)

from lips.physical_simulator.GetfemSimulator.GetfemBricks.BrickBase import BrickBase
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

modelVarByPhyField={
    PFN.displacement:"u",
    PFN.contactMultiplier:"lambda",
    PFN.contactUnilateralMultiplier:"lambda"
}

class SourceFactory(Factory):
    _Catalog = {}
    _SetCatalog = set()
    def __init__(self):
        super(SourceFactory,self).__init__()

def CreateSourceBrick(name,ops=None):
    return SourceFactory.Create(name,ops)

class AddUniformSourceTerm(BrickBase):
    def __init__(self):
        super(AddUniformSourceTerm,self).__init__()
        self._name="Uniform source term Brick"

    def Build(self,problemParams:dict,brickParams:dict):
        sourceTagname,_=brickParams["regionTag"]
        if sourceTagname!="ALL":
            raise Exception("Case not covered yet")
        valNames=["source_x","source_y","source_z"]
        sourceVal=[brickParams[valNames[axeId]] for axeId in range(problemParams["dimension"])]
        model=problemParams["model"]
        model.add_initialized_data('f', sourceVal)
        idBrick=model.add_source_term_brick(problemParams["integrationMethod"], 'u', 'f') 
        return idBrick
SourceFactory.RegisterClass("Uniform",AddUniformSourceTerm)

class AddVariableSourceTerm(BrickBase):
    def __init__(self):
        super(AddVariableSourceTerm,self).__init__()
        self._name="Variable source term Brick"

    def Build(self,problemParams:dict,brickParams:dict):
        sourceTagname,_=brickParams["regionTag"]
        if sourceTagname!="ALL":
            raise Exception("Case not covered yet")
        sourceTerm=brickParams["source_term"]
        axesName=["X","Y","Z"]
        for axeId in range(problemParams["dimension"]):
            axeName=axesName[axeId]
            sourceTerm=sourceTerm.replace( axeName, "X("+str(axeId+1)+")")
        model=problemParams["model"]
        idBrick=model.add_linear_generic_assembly_brick(problemParams["integrationMethod"], "-["+sourceTerm+"].Test_u")
        return idBrick
SourceFactory.RegisterClass("Variable",AddVariableSourceTerm)

class DirichletFactory(Factory):
    _Catalog = {}
    _SetCatalog = set()
    def __init__(self):
        super(DirichletFactory,self).__init__()

def CreateDirichletBrick(name,ops=None):
    return DirichletFactory.Create(name,ops)

#EssentialBoundary
class AddOrientedDirichletCondition(BrickBase):
    def __init__(self):
        super(AddOrientedDirichletCondition,self).__init__()
        self._name="Oriented Dirichlet condition"

    def Build(self,problemParams:dict,brickParams:dict):
        dimension=problemParams["dimension"]

        if dimension==2:
            amplitudeDisp,angleDisp=brickParams["Disp_Amplitude"],brickParams["Disp_Angle"]
            enforcedDisp=[amplitudeDisp*math.cos(angleDisp),
                          amplitudeDisp*math.sin(angleDisp)]
            entriesToRemove=['Disp_Amplitude','Disp_Angle']
        elif dimension==3:
            amplitudeDisp,thetaAngle,phiAngle=brickParams["Amplitude"],brickParams["ThetaAngle"],brickParams["PhiAngle"]
            enforcedDisp=[amplitudeDisp*math.sin(thetaAngle)*math.cos(phiAngle),
                          amplitudeDisp*math.sin(thetaAngle)*math.sin(phiAngle),
                          amplitudeDisp*math.cos(thetaAngle)]
            entriesToRemove=["Amplitude","ThetaAngle","PhiAngle"]

        classicalDirichletBrickParams=brickParams
        for entryToRemove in entriesToRemove:
            classicalDirichletBrickParams.pop(entryToRemove, None)
        if "enforcementCondition" not in classicalDirichletBrickParams.keys():
            classicalDirichletBrickParams["enforcementCondition"]="withMultipliers"
        valNames=["val_x","val_y","val_z"]
        for axeId in range(dimension):
            classicalDirichletBrickParams[valNames[axeId]]=enforcedDisp[axeId]

        classicalDirichletBrick=CreateDirichletBrick(name="GlobalVector")
        idBrick=classicalDirichletBrick(problemParams=problemParams, brickParams=classicalDirichletBrickParams)

        return idBrick
DirichletFactory.RegisterClass("scalar",AddOrientedDirichletCondition)


class AddClassicalDirichletCondition(BrickBase):
    def __init__(self):
        super(AddClassicalDirichletCondition,self).__init__()
        self._name="Global dirichlet condition"

    def Build(self,problemParams:dict,brickParams:dict):
        if "enforcementCondition" not in brickParams:
            brickParams["enforcementCondition"]="withSimplification"

        dirichTagname,dirichId=brickParams["regionTag"]
        dirichZone=problemParams["tagMap"][dirichTagname]

        dirichVariable='DirichletData'+str(dirichId)
        valNames=["val_x","val_y","val_z"]
        dirichVal=[brickParams[valNames[axeId]] for axeId in range(problemParams["dimension"])]
        model=problemParams["model"]
        model.add_initialized_data(dirichVariable, dirichVal)
        if brickParams["enforcementCondition"]=="withSimplification":
            idBrick=model.add_Dirichlet_condition_with_simplification('u', dirichZone, dirichVariable)
        elif brickParams["enforcementCondition"]=="withMultipliers":
            integrationMethod=problemParams["integrationMethod"]
            idBrick = model.add_Dirichlet_condition_with_multipliers(integrationMethod, 'u', 1, dirichZone, dirichVariable)
        else:
            raise Exception("Condition "+brickParams["enforcementCondition"]+" not handled")
        return idBrick
DirichletFactory.RegisterClass("GlobalVector",AddClassicalDirichletCondition)

class AddPointwiseDirichletConditionVector(BrickBase):
    def __init__(self):
        super(AddPointwiseDirichletConditionVector,self).__init__()
        self._name="Pointwise dirichlet condition"

    def Build(self,problemParams:dict,brickParams:dict): 
        model=problemParams["model"]
        _,dirichId=brickParams["regionTag"]
        ptsDisp = brickParams['ptsxy_vector'].T
        model.add_initialized_data('pts_vec'+str(dirichId), ptsDisp)
        enforcedDisp=brickParams['uxy_vector']

        idBricks=[]
        problemDimension=problemParams["dimension"]
        for axeId,axeName in zip(range(problemDimension),["x","y","z"]):
            enforcedDisp_axe = enforcedDisp[:, axeId]
            nbDofPerDimension=enforcedDisp.shape[0]
            dataunitv = np.zeros((problemDimension,nbDofPerDimension))
            dataunitv[axeId,:]=1
            model.add_initialized_data('DData_vec_'+axeName+str(dirichId), enforcedDisp_axe)
            model.add_initialized_data('dataunitv_'+axeName+str(dirichId), dataunitv)
            idBrick = model.add_pointwise_constraints_with_multipliers('u', 'pts_vec'+str(dirichId), 'dataunitv_'+axeName+str(dirichId), 'DData_vec_'+axeName+str(dirichId))
            idBricks.append(idBrick)
        
        return idBricks
DirichletFactory.RegisterClass("vector",AddPointwiseDirichletConditionVector)


def ComputeDirichletConditionRHS(problemParams:dict,brickParams:dict):
    dimension=problemParams["dimension"]
    mfu = problemParams["feSpace"]
    dirichTagname,_=brickParams["regionTag"]
    pts_rom, u_rom = brickParams["ptsxy"], brickParams["uxy"]

    dof = mfu.basic_dof_nodes().T[0::dimension, :]
    dof_hole = mfu.basic_dof_on_region(problemParams["tagMap"][dirichTagname])
    myTree = cKDTree(pts_rom)
    _, inverse_map = myTree.query(dof, k=1)

    u_rom = u_rom[inverse_map, :]
    u_rom_final = np.zeros((dimension * dof.shape[0], ))
    for axeId in range(dimension):
        u_rom_final[axeId::dimension] = u_rom[:,axeId]

    g_rollingRHS = np.zeros((dimension * dof.shape[0],))
    for axeId in range(dimension):
        g_rollingRHS[dof_hole[axeId::dimension]] = u_rom_final[dof_hole[axeId::dimension]]
    return g_rollingRHS

class AddDirichletConditionRHS(BrickBase):
    def __init__(self):
        super(AddDirichletConditionRHS,self).__init__()
        self._name="Dirichlet condition from rhs"

    def Build(self,problemParams:dict,brickParams:dict):
        g_rollingRHS=ComputeDirichletConditionRHS(problemParams=problemParams, brickParams=brickParams)

        dirichTagName,dirichId=brickParams["regionTag"]
        dirichVariable='DirichletData'+str(dirichId)
        model,mfu,mim = problemParams["model"],problemParams["feSpace"],problemParams['integrationMethod']
        model.add_initialized_fem_data(dirichVariable, mfu, g_rollingRHS)
        dirichZone=problemParams["tagMap"][dirichTagName]
        idBrick = model.add_Dirichlet_condition_with_multipliers(mim, 'u', 1, dirichZone, dirichVariable)
        return idBrick

DirichletFactory.RegisterClass("rhs",AddDirichletConditionRHS)

#Neumann conditions
class NeumannFactory(Factory):
    _Catalog = {}
    _SetCatalog = set()
    def __init__(self):
        super(NeumannFactory,self).__init__()

def CreateNeumannBrick(name,ops=None):
    return NeumannFactory.Create(name,ops)

class AddNeumannCondition(BrickBase):
    def __init__(self):
        super(AddNeumannCondition,self).__init__()
        self._name="Standard neumann condition"

    def Build(self,problemParams:dict,brickParams:dict):
        model,mim = problemParams["model"],problemParams["integrationMethod"]
        neumannTagname,neumannId=brickParams["regionTag"]
        neumannZone=problemParams["tagMap"][neumannTagname]
        valNames=["fx","fy","fz"]
        neumannVal=[brickParams[valNames[axeId]] for axeId in range(problemParams["dimension"])]
    
        model.add_initialized_data('NeumannData'+str(neumannId), neumannVal)
        idBrick=model.add_source_term_brick(mim, 'u', 'NeumannData'+str(neumannId),neumannZone)
        return idBrick
NeumannFactory.RegisterClass("StandardNeumann",AddNeumannCondition)

class AddRimRigidityNeumannCondition(BrickBase):
    def __init__(self):
        super(AddRimRigidityNeumannCondition,self).__init__()
        self._name="Rim rigidity neumann condition"

    def Build(self,problemParams:dict,brickParams:dict):
        model,mfl,mim = problemParams["model"],problemParams["feSpace"],problemParams["integrationMethod"]
        neumannTagname,neumannId=brickParams["regionTag"]
        neumannZone=problemParams["tagMap"][neumannTagname]

        pressure=[brickParams["Force"]/(8*2*np.pi)]
        model.add_filtered_fem_variable('lambda_D',mfl, neumannZone)
        neumannVariable='F'+str(neumannId)
        model.add_initialized_data(neumannVariable, pressure)
        model.add_variable('alpha_D', 1)

        normalFoundation=problemParams["dimension"]*[0.0]
        normalFoundation[1]=1
        model.add_initialized_data("normalFoundation", [normalFoundation])
        idBrick=model.add_linear_term(mim,\
            '-lambda_D.Test_u + (alpha_D*normalFoundation - u).Test_lambda_D + (lambda_D.normalFoundation + '+neumannVariable+')*Test_alpha_D + 1E-6*alpha_D*Test_alpha_D', neumannZone)
        return idBrick
NeumannFactory.RegisterClass("RimRigidityNeumann",AddRimRigidityNeumannCondition)

#Contact Conditions
class MultiplierRequiredConditionLookUp():
    _Catalog = dict()
    _SetCatalog = set()

    @classmethod
    def RegisterConditionEnforcedMethod(cls, conditionName:str, multRequired:bool, withError = True):
        if conditionName in cls._Catalog and withError:
           raise (Exception ("Contact condition "+ str(conditionName) +" already in the catalog") )
        cls._Catalog[conditionName] = multRequired
        cls._SetCatalog.add( (conditionName,multRequired))

    @classmethod
    def AllEntries(cls):
        return cls._SetCatalog

    @classmethod
    def MultiplierVariableRequired(cls,conditionName):
        return cls._Catalog[conditionName]


class ContactFactory(Factory):
    _Catalog = {}
    _SetCatalog = set()
    def __init__(self):
        super(ContactFactory,self).__init__()

def CreateContactBrick(name,ops=None):
    return ContactFactory.Create(name,ops)

class AddUnilatContactWithFric(BrickBase):
    def __init__(self):
        super(AddUnilatContactWithFric,self).__init__()
        self._name="Plane Unilateral contact with friction enforced with multipliers"

    def Build(self,problemParams:dict,brickParams:dict):
        model,mim = problemParams["model"],problemParams["integrationMethod"]
        contactTagname,_=brickParams["regionTag"]
        contactZone=problemParams["tagMap"][contactTagname]
        if "augmentationParam" not in brickParams.keys():
            r=0.2
        else:
            r=brickParams["augmentationParam"]
        model.add_initialized_data('r', [r])
        normal=problemParams["dimension"]*[0.0]
        normal[1]=-1
        model.add_initialized_data('N1', normal)
        model.add_initialized_data('gap', [brickParams["gap"]])
        model.add_initialized_data('fric', [brickParams["fricCoeff"]])
        idBrick1=model.add_linear_generic_assembly_brick(mim, '-lambda.(Test_u)', contactZone)
        idBrick2=model.add_linear_generic_assembly_brick(mim, '-(1/r)*lambda.Test_lambda', contactZone)
        idBrick3=model.add_nonlinear_generic_assembly_brick(mim, '(1/r)*Coulomb_friction_coupled_projection(lambda, N1, u,X(2)+gap-u.N1, fric, r).Test_lambda', contactZone)
        return idBrick1,idBrick2,idBrick3
ContactFactory.RegisterClass("Plane",AddUnilatContactWithFric)
MultiplierRequiredConditionLookUp.RegisterConditionEnforcedMethod("Plane",PFN.contactMultiplier)

class AddInclinedUnilatContactWithFric(BrickBase):
    def __init__(self):
        super(AddInclinedUnilatContactWithFric,self).__init__()
        self._name="Inclined plane unilateral contact with friction enforced with multipliers"

    def Build(self,problemParams:dict,brickParams:dict):
        if problemParams["dimension"]!=2:
            raise Exception(self._name+" condition only working in 2D!")
        found_angle=brickParams["Found_angle"]
        shift=15.0
        if "augmentationParam" not in brickParams.keys():
            r=0.2
        else:
            r=brickParams["augmentationParam"]
        model,mim = problemParams["model"],problemParams["integrationMethod"]
        contactTagname,_=brickParams["regionTag"]
        contactZone=problemParams["tagMap"][contactTagname]
        model.add_initialized_data('r', [r])
        model.add_initialized_data('N1', [math.cos(found_angle),math.sin(found_angle)])
        model.add_initialized_data('shiftDist', [shift*(1+math.sin(found_angle))])
        model.add_initialized_data('fric', [brickParams["fricCoeff"]])
        idBrick1=model.add_linear_generic_assembly_brick(mim, '-lambda.(Test_u)', contactZone)
        idBrick2=model.add_linear_generic_assembly_brick(mim, '-(1/r)*lambda.Test_lambda', contactZone)
        idBrick3=model.add_nonlinear_generic_assembly_brick(mim, '(1/r)*Coulomb_friction_coupled_projection(lambda, N1, u,-(X.N1-shiftDist)-u.N1, fric, r).Test_lambda', contactZone)
        return idBrick1,idBrick2,idBrick3
ContactFactory.RegisterClass("Inclined",AddInclinedUnilatContactWithFric)
MultiplierRequiredConditionLookUp.RegisterConditionEnforcedMethod("Inclined",PFN.contactMultiplier)

class AddUnilatContact(BrickBase):
    def __init__(self):
        super(AddUnilatContact,self).__init__()
        self._name="Plane unilateral contact without friction enforced with multipliers"

    def Build(self,problemParams:dict,brickParams:dict):
        if "augmentationParam" not in brickParams.keys():
            r=0.2
        else:
            r=brickParams["augmentationParam"]
        model,mim = problemParams["model"],problemParams["integrationMethod"]
        contactTagname,_=brickParams["regionTag"]
        contactZone=problemParams["tagMap"][contactTagname]
        model.add_initialized_data('r', [r])
        normal=problemParams["dimension"]*[0.0]
        normal[1]=-1
        model.add_initialized_data('N1', normal)
        model.add_initialized_data('gap', [brickParams["gap"]])
        idBrick1=model.add_linear_generic_assembly_brick(mim, 'lambda*(Test_u.N1)', contactZone)
        idBrick2=model.add_nonlinear_generic_assembly_brick(mim, '-(1/r)*(-lambda + neg_part(-lambda-(r)*(u.N1-gap-X(2))))*Test_lambda', contactZone);
        return idBrick1,idBrick2
ContactFactory.RegisterClass("NoFriction",AddUnilatContact)
MultiplierRequiredConditionLookUp.RegisterConditionEnforcedMethod("NoFriction",PFN.contactUnilateralMultiplier)

class AddPenalizedUnilatContactWithFric(BrickBase):
    def __init__(self):
        super(AddPenalizedUnilatContactWithFric,self).__init__()
        self._name="Plane unilateral contact without friction enforced with penalization"

    def Build(self,problemParams:dict,brickParams:dict):
        if "PenalizationParam" not in brickParams.keys():
            r=10
        else:
            r=brickParams["PenalizationParam"]
        model,mim = problemParams["model"],problemParams["integrationMethod"]
        contactTagname,_=brickParams["regionTag"]
        contactZone=problemParams["tagMap"][contactTagname]
        model.add_initialized_data('r', [r])
        model.add_initialized_data('fric', [brickParams["fricCoeff"]])
        mfObstacle=problemParams["feSpace"]
        horizontalObstacle = mfObstacle.eval("y")
        model.add_initialized_fem_data('obstacle', mfObstacle, horizontalObstacle)
        model.add_penalized_contact_with_rigid_obstacle_brick(mim, 'u', 'obstacle', 'r', 'fric', contactZone)
ContactFactory.RegisterClass("PlanePenalized",AddPenalizedUnilatContactWithFric)
MultiplierRequiredConditionLookUp.RegisterConditionEnforcedMethod("PlanePenalized",None)
