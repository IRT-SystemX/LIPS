#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod

from BasicTools.Helpers.Factory import Factory

import getfem as gf
gf.util_trace_level(1)

from lips.physical_simulator.GetfemSimulator.GetfemBricks.BrickBase import BrickBase
from lips.physical_simulator.GetfemSimulator.GetfemBricks.FeSpaces import DefineDiscontinuousFESpace
from lips.physical_simulator.GetfemSimulator.GetfemBricks.Utilities import ComputeIntegralOverBoundary

def AddFiniteStrainMacroToModel(model):
    model.add_macro("F", "Grad_u+Id(2)")
    model.add_macro("J", "Det(F)")
    model.add_macro("epsilon", "(Grad_u+(Grad_u)')/2")
    model.add_macro("Green_Lagrange_E", "epsilon+ (Grad_u)'*Grad_u/2")

def ComputeVonMises(model,material,mesh,integMethod):
    mfvm=DefineDiscontinuousFESpace(mesh=mesh,elements_degree=1,dof=1)
    _,materialBlock=material
    behaviourBrick=CreateBehaviourLawBrick(materialBlock["law"])
    vonMisesExpression=behaviourBrick.GetVonMisesExpression(problemDimension=mesh.dim())
    vonMises=model.local_projection(integMethod,vonMisesExpression,mfvm)
    return vonMises

def ComputeEquilibriumResidual(model,material,mim):
    _,materialBlock=material
    if materialBlock["law"]=="LinearElasticity":
        div_sigma_x=gf.asm_generic(mim=mim, order=0, expression="Norm_sqr( Trace(Grad(clambda*Trace(epsilon)*Id(2)+2*cmu*epsilon)(1,:,:)) + f(1))", model=model,region=-1)
        div_sigma_y=gf.asm_generic(mim=mim, order=0, expression="Norm_sqr( Trace(Grad(clambda*Trace(epsilon)*Id(2)+2*cmu*epsilon)(2,:,:)) + f(2))", model=model,region=-1)
        div_sigma=[div_sigma_x,div_sigma_y]
    else:
        raise Exception("Not implemented yet")
    return div_sigma

def ComputeTotalElasticEnergy(model,behaviour_type,integMethod,region=-1):
    behaviourBrick=CreateBehaviourLawBrick(behaviour_type)
    strainEnergyExpression=behaviourBrick.GetStrainEnergyExpression(problemDimension=2)
    return ComputeIntegralOverBoundary(model=model,expression=strainEnergyExpression,mim=integMethod,region=region)

class BehaviourLawFactory(Factory):
    _Catalog = {}
    _SetCatalog = set()
    def __init__(self):
        super(BehaviourLawFactory,self).__init__()

def CreateBehaviourLawBrick(name,ops=None):
    return BehaviourLawFactory.Create(name,ops)


class BehaviourBrickBase(BrickBase):

    @classmethod
    @abstractmethod
    def GetVonMisesExpression(cls,problemDimension):
        pass

    @classmethod
    @abstractmethod
    def GetStrainEnergyExpression(cls,modelParams,modelVar,problemDimension):
        pass

class AddLinearElasticity(BehaviourBrickBase):
    modelParamMapping={"clambda":"clambda","cmu":"cmu"}
    modelVar="u"

    def __init__(self):
        super(AddLinearElasticity,self).__init__()
        self._name="Linear elasticity: Hooke's law"

    @classmethod
    def GetVonMisesExpression(cls,problemDimension=None):
        vonMisesExpression="sqrt(1.5)*Norm(Deviator(cmu*(Grad_u+Grad_u')))"
        return vonMisesExpression

    @classmethod
    def GetStrainEnergyExpression(cls):
        modelParams=cls.modelParamMapping
        strainEnergyExpression="Trace(("+modelParams["clambda"]+"*Trace(epsilon)*Id(2) + 2*"+modelParams["cmu"]+"*epsilon)'*epsilon)/2"
        return strainEnergyExpression

    def Build(self,problemParams:dict,brickParams:dict):
        model,integMethod = problemParams["model"],problemParams["integrationMethod"]
        materialTagname,_=brickParams["regionTag"]
        region,suffixParam=GetRegionInformations(materialTagname,problemParams["tagMap"])
        planeStress=(problemParams["dimension"]==2)
        clambda,cmu = ComputeLameCoeff(young=brickParams["young"],poisson=brickParams["poisson"],planeStress=planeStress)
        model.add_initialized_data('cmu'+str(suffixParam), [cmu])
        model.add_initialized_data('clambda'+str(suffixParam), [clambda])
        idBrick=model.add_linear_generic_assembly_brick(integMethod, "clambda*(Div_u*Div_Test_u)+cmu*((Grad_u+(Grad_u)'):Grad_Test_u)",region)
        return idBrick
BehaviourLawFactory.RegisterClass("LinearElasticity",AddLinearElasticity)

class AddIncompMooneyRivlin(BehaviourBrickBase):
    modelParamMapping={"c1":"paramsIMR(1)","c2":"paramsIMR(2)"}
    modelVar="u"
    def __init__(self):
        super(AddIncompMooneyRivlin,self).__init__()
        self._name="Nonlinear elasticity: incompressible mooney rivlin"

    @classmethod
    def GetVonMisesExpression(cls,problemDimension):
        if problemDimension==2:
            vonMisesExpression="sqrt(1.5)*Norm(Deviator(Cauchy_stress_from_PK2(Plane_Strain_Incompressible_Mooney_Rivlin_PK2(Grad_u, [paramsIMR(1);paramsIMR(2)]),Grad_u)))"
        elif problemDimension==3:
            vonMisesExpression="sqrt(1.5)*Norm(Deviator(Cauchy_stress_from_PK2(Incompressible_Mooney_Rivlin_PK2(Grad_u, [paramsIMR(1);paramsIMR(2)]),Grad_u)))"   
        else:
            raise Exception("Only dimension 2 and 3 are handled")   
        return vonMisesExpression

    @classmethod
    def GetStrainEnergyExpression(cls,problemDimension):
        modelVar,modelParams=cls.modelVar,cls.modelParamMapping
        if problemDimension==2:
            strainEnergyExpression="Plane_Strain_Incompressible_Mooney_Rivlin_potential(Grad_"+modelVar+", ["+modelParams["c1"]+";"+modelParams["c2"]+"])"
        elif problemDimension==3:
            strainEnergyExpression="Incompressible_Mooney_Rivlin_potential(Grad_"+modelVar+", ["+modelParams["c1"]+";"+modelParams["c2"]+"])"
        else:
            raise Exception("Only dimension 2 and 3 are handled")
        return strainEnergyExpression

    def Build(self,problemParams:dict,brickParams:dict):
        model,integMethod = problemParams["model"],problemParams["integrationMethod"]
        materialTagname,_=brickParams["regionTag"]
        region,suffixParam=GetRegionInformations(materialTagname,problemParams["tagMap"])
        model.add_initialized_data('paramsIMR'+str(suffixParam), [brickParams["MooneyRivlinC1"],brickParams["MooneyRivlinC2"]])
        expression=self.GetStrainEnergyExpression(problemDimension=problemParams["dimension"])
        idBrick=model.add_nonlinear_generic_assembly_brick(integMethod,expression,region)
        return idBrick
BehaviourLawFactory.RegisterClass("IncompressibleMooneyRivlin",AddIncompMooneyRivlin)


def AddIncompressibilityBrick(model,mim):
    idBrick=model.add_finite_strain_incompressibility_brick(mim, 'u', 'p')
    return idBrick

class AddCompressibleNeoHookean(BehaviourBrickBase):
    modelParamMapping={"clambda":"clambda","cmu":"cmu"}
    modelVar="u"
    def __init__(self):
        super(AddCompressibleNeoHookean,self).__init__()
        self._name="Nonlinear elasticity: compressible neo hookean"

    @classmethod
    def GetVonMisesExpression(cls,problemDimension):
        raise Exception("Not implemented yet, sorry")

    @classmethod
    def GetStrainEnergyExpression(cls,problemDimension=None):
        modelParams=cls.modelParamMapping
        strainEnergyExpression="("+modelParams["cmu"]+"/2)* ( Trace(Right_Cauchy_Green(F)) - 2 - 2*log(J) )+ ("+modelParams["clambda"]+"/2)* pow(2*log(J),2) "
        return strainEnergyExpression

    def Build(self,problemParams:dict,brickParams:dict):
        model,integMethod = problemParams["model"],problemParams["integrationMethod"]
        materialTagname,_=brickParams["regionTag"]
        region,suffixParam=GetRegionInformations(materialTagname,problemParams["tagMap"])
        planeStress=(problemParams["dimension"]==2)
        clambda,cmu = ComputeLameCoeff(young=brickParams["young"],poisson=brickParams["poisson"],planeStress=planeStress)
        model.add_initialized_data('cmu'+str(suffixParam), [cmu])
        model.add_initialized_data('clambda'+str(suffixParam), [clambda])
        expression=self.GetStrainEnergyExpression(problemDimension=problemParams["dimension"])
        idBrick=model.add_nonlinear_generic_assembly_brick(integMethod,expression,region)
        return idBrick
BehaviourLawFactory.RegisterClass("CompressibleNeoHookean",AddCompressibleNeoHookean)

class AddSaintVenantKirchhoff(BehaviourBrickBase):
    modelParamMapping={"clambda":"clambda","cmu":"cmu"}
    modelVar="u"
    def __init__(self):
        super(AddSaintVenantKirchhoff,self).__init__()
        self._name="Nonlinear elasticity: Saint-venant Kirchhoff"

    @classmethod
    def GetVonMisesExpression(cls,problemDimension):
        if problemDimension==2:
            vonMisesExpression="sqrt(1.5)*Norm(Deviator(Cauchy_stress_from_PK2(Plane_Strain_Saint_Venant_Kirchhoff_PK2(Grad_u, [clambda; cmu]),Grad_u)))"
        elif problemDimension==3:
            vonMisesExpression="sqrt(1.5)*Norm(Deviator(Cauchy_stress_from_PK2(Saint_Venant_Kirchhoff_PK2(Grad_u, [clambda; cmu]),Grad_u)))"
        else:
            raise Exception("Only dimension 2 and 3 are handled") 
        return vonMisesExpression

    @classmethod
    def GetStrainEnergyExpression(cls,problemDimension):
        modelVar,modelParams=cls.modelVar,cls.modelParamMapping
        if problemDimension==2:
            strainEnergyExpression="Plane_Strain_Saint_Venant_Kirchhoff_potential(Grad_"+modelVar+", ["+modelParams["clambda"]+"; "+modelParams["cmu"]+"])"
        elif problemDimension==3:
            strainEnergyExpression="Saint_Venant_Kirchhoff_potential(Grad_"+modelVar+", ["+modelParams["clambda"]+"; "+modelParams["cmu"]+"])"
        else:
            raise Exception("Only dimension 2 and 3 are handled")
        return strainEnergyExpression

    def Build(self,problemParams:dict,brickParams:dict):
        model,integMethod = problemParams["model"],problemParams["integrationMethod"]
        materialTagname,_=brickParams["regionTag"]
        region,suffixParam=GetRegionInformations(materialTagname,problemParams["tagMap"])
        planeStress=(problemParams["dimension"]==2)
        clambda,cmu = ComputeLameCoeff(young=brickParams["young"],poisson=brickParams["poisson"],planeStress=planeStress)
        model.add_initialized_data('cmu'+str(suffixParam), [cmu])
        model.add_initialized_data('clambda'+str(suffixParam), [clambda])
        expression=self.GetStrainEnergyExpression(problemDimension=problemParams["dimension"])
        idBrick=model.add_nonlinear_generic_assembly_brick(integMethod,expression,region)
        return idBrick
BehaviourLawFactory.RegisterClass("SaintVenantKirchhoff",AddSaintVenantKirchhoff)


def ComputeLameCoeff(young,poisson,planeStress=False):
    clambda = young*poisson/((1+poisson)*(1-2*poisson)) 
    cmu = young/(2*(1+poisson)) 
    if planeStress:
        clambda,cmu = ApplyPlaneStressTransformation(clambda,cmu)              
    return clambda,cmu

def ApplyPlaneStressTransformation(clambda,cmu):
    return 2*clambda*cmu/(clambda+2*cmu),cmu

def GetRegionInformations(materialTagname,tagMap):
    if materialTagname=="ALL":
        region=-1
        suffixParam=""
    else:
        region=tagMap[materialTagname]
        suffixParam=materialTagname
    return region,suffixParam
