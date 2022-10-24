#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

from BasicTools.Helpers.Factory import Factory

from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemMecaProblem
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
import lips.physical_simulator.GetfemSimulator.GetfemBricks.ModelTools as gfModel
import lips.physical_simulator.GetfemSimulator.GetfemBricks.MeshTools as gfMeshTools
import lips.physical_simulator.GetfemSimulator.GetfemBricks.BehaviourLaw as gfBehaviour
from lips.physical_simulator.GetfemSimulator.GetfemBricks.Utilities import ComputeIntegralOverBoundary

class PhysicalCriteriaFactory(Factory):
    _Catalog = {}
    _SetCatalog = set()
    def __init__(self):
        super(PhysicalCriteriaFactory,self).__init__()

def CreatePhysicalCriteria(name,ops=None):
    return PhysicalCriteriaFactory.Create(name,ops)

class PhysicalCriteriaBase(metaclass=abc.ABCMeta):
    def __init__(self,problem):
        self.model=problem.model
        self.integrMethods=problem.integrMethods
        self.requiredFields=[]
        self.physicalInputDependancy=False

    def AllRequiredFieldForCriteria(self,fieldNames):
        if not self.requiredFields:
            print("Warning: this criteria does not require any field to be computed")
            return True
        return set(self.requiredFields)<=set(fieldNames)

    def SetExternalSolutions(self,fields):
        if self.AllRequiredFieldForCriteria(fields.keys()):
            for fieldName,fieldValue in fields.items():
                modelVarName=gfModel.modelVarByPhyField[fieldName]
                gfModel.SetModelVariableValue(self.model,modelVarName,fieldValue)
        else:
            missingFields=list(set(self.requiredFields) - set(fields.keys()))
            missingFields=', '.join("{}".format(k) for k in missingFields)
            raise Exception("Criteria cannot be computed with this solution; fields missing:", missingFields)

    @abc.abstractmethod
    def ComputeValue(self):
        pass

def DeformedVolumeBuilder(problem:GetfemMecaProblem):
    res = DeformedVolume(problem)
    return res

class DeformedVolume(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(DeformedVolume,self).__init__(problem=problem)
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        return ComputeIntegralOverBoundary(model=self.model,expression="Det(Id(2)+Grad_u)",mim=self.integrMethods["standard"])
PhysicalCriteriaFactory.RegisterClass("DeformedVolume",DeformedVolume,DeformedVolumeBuilder)

def UnilateralContactPressureBuilder(problem:GetfemMecaProblem):
    res = UnilateralContactPressure(problem)
    return res

class UnilateralContactPressure(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(UnilateralContactPressure,self).__init__(problem=problem)
        if not problem.contact:
            raise Exception("Cannot use this criteria if there is no contact condition")
        self.refNumByRegion=problem.refNumByRegion
        self.requiredFields=[PFN.contactUnilateralMultiplier]

    def ComputeValue(self,contactBoundary):
        return ComputeIntegralOverBoundary(model=self.model,expression="lambda",mim=self.integrMethods["composite"],region=self.refNumByRegion[contactBoundary])
PhysicalCriteriaFactory.RegisterClass("UnilateralContactPressure",UnilateralContactPressure,UnilateralContactPressureBuilder)

def FrictionalContactPressureBuilder(problem:GetfemMecaProblem):
    res = FrictionalContactPressure(problem)
    return res

class FrictionalContactPressure(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(FrictionalContactPressure,self).__init__(problem=problem)
        if not problem.contact:
            raise Exception("Cannot use this criteria if there is no contact condition")
        self.refNumByRegion=problem.refNumByRegion
        self.requiredFields=[PFN.contactMultiplier]

    def ComputeValue(self,contactBoundary):
        return ComputeIntegralOverBoundary(model=self.model,expression="lambda",mim=self.integrMethods["composite"],region=self.refNumByRegion[contactBoundary])
PhysicalCriteriaFactory.RegisterClass("FrictionalContactPressure",FrictionalContactPressure,FrictionalContactPressureBuilder)

def TotalElasticEnergyBuilder(problem:GetfemMecaProblem):
    res = TotalElasticEnergy(problem)
    return res

class TotalElasticEnergy(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(TotalElasticEnergy,self).__init__(problem=problem)
        self.refNumByRegion=problem.refNumByRegion
        self.materials=problem.materials
        self.requiredFields=[PFN.displacement]
        self.physicalInputDependancy=True

    def ComputeValue(self):
        elasticEnergy=0
        if len(self.materials)>1:
            raise Exception("Have only tested for one material for now")

        for tagname,material in self.materials:
            if tagname=="ALL":
                materialLaw=material["law"]
                elasticEnergy+=gfBehaviour.ComputeTotalElasticEnergy(model=self.model,
                                                                behaviour_type=materialLaw,
                                                                integMethod=self.integrMethods["standard"])
                break
            else:
                materialLaw=material["law"]
                elasticEnergy+=gfBehaviour.ComputeTotalElasticEnergy(model=self.model,
                                                                behaviour_type=materialLaw,
                                                                integMethod=self.integrMethods["standard"],
                                                                region=self.refNumByRegion[tagname])
        return elasticEnergy
PhysicalCriteriaFactory.RegisterClass("TotalElasticEnergy",TotalElasticEnergy,TotalElasticEnergyBuilder)

def MaxVonMisesBuilder(problem:GetfemMecaProblem):
    res = MaxVonMises(problem)
    return res

class MaxVonMises(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(MaxVonMises,self).__init__(problem=problem)
        self.materials=problem.materials
        self.mesh=problem.mesh
        self.requiredFields=[PFN.displacement]
        self.physicalInputDependancy=True


    def ComputeValue(self):
        if len(self.materials)>1:
            raise Exception("Multi material not handled yet for this criteria!")

        vonMisesField=gfBehaviour.ComputeVonMises(model=self.model,
                                            material=self.materials[0],
                                            integMethod=self.integrMethods["standard"],
                                            mesh=self.mesh)
        return np.max(vonMisesField)
PhysicalCriteriaFactory.RegisterClass("MaxVonMises",MaxVonMises,MaxVonMisesBuilder)

def MaxDispBuilder(problem:GetfemMecaProblem):
    res = MaxDisp(problem)
    return res

class MaxDisp(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(MaxDisp,self).__init__(problem=problem)
        self.mesh=problem.mesh
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        meshDimension=self.mesh.dim()
        displacements=gfModel.GetModelVariableValue(self.model,"u")

        if displacements.shape[0] % meshDimension:
            raise Exception("Displacement size unconsistent with mesh dimension")

        fieldSizePerDof=displacements.shape[0]//meshDimension
        displacements_d=np.zeros((fieldSizePerDof,meshDimension))
        for dof in range(meshDimension):
            displacements_d[:,dof]=displacements[dof::meshDimension]
        return np.max(displacements_d,axis=0)
PhysicalCriteriaFactory.RegisterClass("MaxDisp",MaxDisp,MaxDispBuilder)

def PhysicalComplianceEquilibriumBuilder(problem:GetfemMecaProblem):
    res = PhysicalComplianceEquilibrium(problem)
    return res

class PhysicalComplianceEquilibrium(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(PhysicalComplianceEquilibrium,self).__init__(problem=problem)
        self.materials=problem.materials
        self.mesh=problem.mesh
        self.requiredFields=[PFN.displacement]
        self.physicalInputDependancy=True

    def ComputeValue(self):
        if len(self.materials)>1:
            raise Exception("Multi material not handled yet for this criteria!")

        equilibriumResidual=gfBehaviour.ComputeEquilibriumResidual(model=self.model,
                                            material=self.materials[0],
                                            mim=self.integrMethods["standard"])
        return equilibriumResidual
PhysicalCriteriaFactory.RegisterClass("PhysicalComplianceEquilibrium",PhysicalComplianceEquilibrium,PhysicalComplianceEquilibriumBuilder)


def ContactMaxPenetrationBuilder(problem:GetfemMecaProblem):
    res = ContactMaxPenetration(problem)
    return res

class ContactMaxPenetration(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(ContactMaxPenetration,self).__init__(problem=problem)
        if not problem.contact:
            raise Exception("Cannot use this criteria if there is no contact condition")
        self.mesh=problem.mesh
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        penetration=gfMeshTools.InterpolateFieldOnMesh(model=self.model,
                                            fieldExpression="u.N1-gap-X(2)",
                                            mesh=self.mesh)
        return np.max(penetration)
PhysicalCriteriaFactory.RegisterClass("ContactMaxPenetration",ContactMaxPenetration,ContactMaxPenetrationBuilder)

def ContactAreaBuilder(problem:GetfemMecaProblem):
    res = ContactArea(problem)
    return res

class ContactArea(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(ContactArea,self).__init__(problem=problem)
        if not problem.contact:
            raise Exception("Cannot use this criteria if there is no contact condition")
        self.mesh=problem.mesh
        self.refNumByRegion=problem.refNumByRegion
        self.requiredFields=[PFN.contactMultiplier]

    def ComputeValue(self,contactBoundary):
        meshDimension=self.mesh.dim()
        contactStress=gfModel.GetModelVariableValue(self.model,"lambda")
        contactStress=contactStress.reshape((contactStress.shape[0]//meshDimension,meshDimension))
        normal=np.array([0,1])
        normal=np.tile(normal,contactStress.shape[0])
        normal=normal.reshape((normal.shape[0]//meshDimension,meshDimension))
        normalContactStress=np.einsum("ij,ij->i",contactStress,normal)
        # scalerMult=np.std(normalMultipliers)
        scalerMult=np.max(normalContactStress)/1e3
        gfModel.AddInitializedData(self.model,"scalerMult",scalerMult)
        return ComputeIntegralOverBoundary(model=self.model,expression="Heaviside(-lambda.N1-scalerMult)",mim=self.integrMethods["composite"],region=self.refNumByRegion[contactBoundary])
PhysicalCriteriaFactory.RegisterClass("ContactArea",ContactArea,ContactAreaBuilder)

def CoulombConsistencyBuilder(problem:GetfemMecaProblem):
    res = CoulombConsistency(problem)
    return res

class CoulombConsistency(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(CoulombConsistency,self).__init__(problem=problem)
        if not problem.contact:
            raise Exception("Cannot use this criteria if there is no contact condition")
        elif len(problem.contact)>1:
            raise Exception("Cannot handle more than one contact condition")
        self.mesh=problem.mesh
        self.refNumByRegion=problem.refNumByRegion
        self.frictionCoeff=problem.contact[0][1]["fricCoeff"]
        self.requiredFields=[PFN.contactMultiplier]
        self.physicalInputDependancy=True

    def ComputeValue(self):
        meshDimension=self.mesh.dim()
        contactStress=gfModel.GetModelVariableValue(self.model,"lambda")
        contactStress=contactStress.reshape((contactStress.shape[0]//meshDimension,meshDimension))
        normal=np.array([0,1])
        normal=np.tile(normal,contactStress.shape[0])
        normal=normal.reshape((normal.shape[0]//meshDimension,meshDimension))
        normalContactStress=np.einsum("ij,ij->i",contactStress,normal)
        tangentialstress=contactStress-np.einsum("ij,i->ij",normal,normalContactStress)
        coulombCriteria=self.frictionCoeff*np.abs(normalContactStress) -np.linalg.norm(tangentialstress,axis=1)
        return np.min(coulombCriteria)
PhysicalCriteriaFactory.RegisterClass("CoulombConsistency",CoulombConsistency,CoulombConsistencyBuilder)

#################################Test#################################
import numpy.testing as npt
import numpy as np

from lips.physical_simulator.GetfemSimulator.GetfemBricks.MeshTools import GenerateSimpleMesh

def CheckIntegrity_NoFrictionContactProblem():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"SaintVenantKirchhoff","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "GlobalVector", "val_x":0.0, "val_y":0.0}] ]
    beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25}] ]
    beamProblem.contact=[ ["BOTTOM",{"type" : "NoFriction","gap":0.0}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    multVariable=("lambda",PFN.contactUnilateralMultiplier,"BOTTOM") 
    beamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    beamProblem.BuildModel()
    beamProblem.RunProblem()
    extSol={PFN.contactUnilateralMultiplier:beamProblem.GetSolution(PFN.contactUnilateralMultiplier),
            PFN.displacement:beamProblem.GetSolution(PFN.displacement)}

    unilatCriteria=CreatePhysicalCriteria("UnilateralContactPressure",beamProblem)
    oriUnilatPressure=unilatCriteria.ComputeValue(contactBoundary="BOTTOM")

    elasCriteria=CreatePhysicalCriteria("TotalElasticEnergy",beamProblem)
    oriElasticEnergy=elasCriteria.ComputeValue()

    beamProblemCopy=type(beamProblem)(other=beamProblem)
    beamProblemCopy.BuildModel()

    unilatCriteriaCopy=CreatePhysicalCriteria("UnilateralContactPressure",beamProblemCopy)
    unilatCriteriaCopy.SetExternalSolutions(extSol)
    newUnilatPressure=unilatCriteriaCopy.ComputeValue(contactBoundary="BOTTOM")
    npt.assert_almost_equal(oriUnilatPressure,newUnilatPressure)

    elasCriteriaCopy=CreatePhysicalCriteria("TotalElasticEnergy",beamProblem)
    elasCriteriaCopy.SetExternalSolutions(extSol)
    newElasticEnergy=elasCriteriaCopy.ComputeValue()
    npt.assert_almost_equal(oriElasticEnergy,newElasticEnergy)
    return "ok"

def CheckIntegrity_FrictionalContactProblem():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "GlobalVector", "enforcementCondition":"withMultipliers","val_x":0.0, "val_y":0.0}] ]
    beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    beamProblem.contact=[ ["BOTTOM",{"type" : "Plane","gap":0.0,"fricCoeff":0.9}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    multVariable=("lambda",PFN.contactMultiplier,"BOTTOM") 
    beamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    beamProblem.BuildModel()
    solverOptions={'lsolver':'MUMPS',
	'lsearch': 'simplest', 'alpha max ratio': 1.5, 'alpha min': 0.2, 'alpha mult': 0.6}
    beamProblem.RunProblem(options=solverOptions)
    extSol={PFN.contactMultiplier:beamProblem.GetSolution(PFN.contactMultiplier),
            PFN.displacement:beamProblem.GetSolution(PFN.displacement)}

    frictCriteria=CreatePhysicalCriteria("FrictionalContactPressure",beamProblem)
    orifrictPressure=frictCriteria.ComputeValue(contactBoundary="BOTTOM")
    beamProblemCopy=type(beamProblem)(other=beamProblem)
    beamProblemCopy.BuildModel()

    frictCriteriaCopy=CreatePhysicalCriteria("FrictionalContactPressure",beamProblemCopy)
    frictCriteriaCopy.SetExternalSolutions(extSol)
    newfrictCriteria=frictCriteriaCopy.ComputeValue(contactBoundary="BOTTOM")
    npt.assert_almost_equal(orifrictPressure,newfrictCriteria)

    return "ok"

def CheckIntegrity_PureDisplacement():
    material1=[["ALL", {"law":"SaintVenantKirchhoff","young":5e5,"poisson":0.3} ]]
    material2=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    # material3=[["ALL", {"law":"IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2":1} ]]
    materialsToTest=[material1,material2]

    for material in materialsToTest:
        beamProblem=GetfemMecaProblem()
        beamProblem.mesh,beamProblem.refNumByRegion=GenerateSimpleMesh(meshSize=20.0)
        beamProblem.materials=material
        beamProblem.dirichlet=[["LEFT",{"type" : "GlobalVector", "val_x":0.0, "val_y":0.0}] ]
        beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-0.5}] ]
        femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
        multVariable=("lambda",PFN.contactUnilateralMultiplier,"BOTTOM") 
        beamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
        beamProblem.BuildModel()
        solverOptions={'lsolver':'MUMPS',
        'lsearch': 'simplest', 'alpha max ratio': 1.5, 'alpha min': 0.2, 'alpha mult': 0.6}
        beamProblem.RunProblem(options=solverOptions)
        extSol={PFN.displacement:beamProblem.GetSolution(PFN.displacement)}

        criteriaToTest=["DeformedVolume","MaxVonMises","MaxDisp"]
        criteriaVal=[None]*len(criteriaToTest)

        for criterionNum,criterionToTest in enumerate(criteriaToTest):
            mycriteria=CreatePhysicalCriteria(criterionToTest,beamProblem)
            criteriaVal[criterionNum]=mycriteria.ComputeValue()

        beamProblemCopy=type(beamProblem)(other=beamProblem)
        beamProblemCopy.BuildModel()

        for criterionNum,criterionToTest in enumerate(criteriaToTest):
            myCopyCriteria=CreatePhysicalCriteria(criterionToTest,beamProblemCopy)
            print(type(myCopyCriteria))
            myCopyCriteria.SetExternalSolutions(extSol)
            newValue=myCopyCriteria.ComputeValue()
            print(newValue)
            npt.assert_almost_equal(criteriaVal[criterionNum],newValue)

    return "ok"


def CheckIntegrity_PhysicalCompliance():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "GlobalVector", "enforcementCondition":"withMultipliers","val_x":0.0, "val_y":0.0}] ]
    beamProblem.sources = [["ALL",{"type" : "Uniform","source_x":0,"source_y":-2e3}] ]
    femVariables={PFN.displacement: {"degree": 2, "dof": 2}}
    multVariable=("lambda",PFN.contactUnilateralMultiplier,"BOTTOM") 
    beamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    beamProblem.BuildModel()
    beamProblem.RunProblem()

    mycriteria=CreatePhysicalCriteria("PhysicalComplianceEquilibrium",beamProblem)
    criteriaVal=mycriteria.ComputeValue()
    print(criteriaVal)

    return "ok"

def BeamFrictionalContactProblem():
    beamProblem=GetfemMecaProblem()
    beamProblem.mesh,beamProblem.refNumByRegion=GenerateSimpleMesh(meshSize=20.0)
    beamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    beamProblem.dirichlet=[["LEFT",{"type" : "GlobalVector", "enforcementCondition":"withMultipliers","val_x":0.0, "val_y":0.0}] ]
    beamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    beamProblem.contact=[ ["BOTTOM",{"type" : "Plane","gap":0.0,"fricCoeff":0.9}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    multVariable=("lambda",PFN.contactMultiplier,"BOTTOM") 
    beamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    beamProblem.BuildModel()
    solverOptions={'lsolver':'MUMPS',
	'lsearch': 'simplest', 'alpha max ratio': 1.5, 'alpha min': 0.2, 'alpha mult': 0.6}

    beamProblem.RunProblem(options=solverOptions)
    return beamProblem

def CheckIntegrity_MaxPenetration():
    beamProblem=BeamFrictionalContactProblem()
    extSol={PFN.displacement:beamProblem.GetSolution(PFN.displacement)}

    mycriteria=CreatePhysicalCriteria("ContactMaxPenetration",beamProblem)
    maxPenetration=mycriteria.ComputeValue()

    beamProblemCopy=type(beamProblem)(other=beamProblem)
    beamProblemCopy.BuildModel()

    unilatCriteriaCopy=CreatePhysicalCriteria("ContactMaxPenetration",beamProblemCopy)
    unilatCriteriaCopy.SetExternalSolutions(extSol)
    newMaxPenetration=unilatCriteriaCopy.ComputeValue()
    npt.assert_almost_equal(maxPenetration,newMaxPenetration)
    return "ok"

def ChechIntegrity_ContactArea():
    beamProblem=BeamFrictionalContactProblem()
    extSol={PFN.contactMultiplier:beamProblem.GetSolution(PFN.contactMultiplier)}

    mycriteria=CreatePhysicalCriteria("ContactArea",beamProblem)
    contactArea=mycriteria.ComputeValue(contactBoundary="BOTTOM")

    beamProblemCopy=type(beamProblem)(other=beamProblem)
    beamProblemCopy.BuildModel()

    unilatCriteriaCopy=CreatePhysicalCriteria("ContactArea",beamProblemCopy)
    unilatCriteriaCopy.SetExternalSolutions(extSol)
    newContactArea=unilatCriteriaCopy.ComputeValue(contactBoundary="BOTTOM")
    npt.assert_almost_equal(contactArea,newContactArea)
    return "ok"    

def CheckIntegrity_CoulombConsistency():
    beamProblem=BeamFrictionalContactProblem()
    extSol={PFN.contactMultiplier:beamProblem.GetSolution(PFN.contactMultiplier)}    

    mycriteria=CreatePhysicalCriteria("CoulombConsistency",beamProblem)
    coulombConsistency=mycriteria.ComputeValue()

    beamProblemCopy=type(beamProblem)(other=beamProblem)
    beamProblemCopy.BuildModel()

    mycriteriaCopy=CreatePhysicalCriteria("CoulombConsistency",beamProblemCopy)
    mycriteriaCopy.SetExternalSolutions(extSol)
    newCoulombConsistency=mycriteriaCopy.ComputeValue()
    npt.assert_almost_equal(coulombConsistency,newCoulombConsistency)
    return "ok"    

def CheckIntegrity():

    totest = [
    CheckIntegrity_NoFrictionContactProblem,
    CheckIntegrity_FrictionalContactProblem,
    CheckIntegrity_PureDisplacement,
    #CheckIntegrity_PhysicalCompliance,
    CheckIntegrity_MaxPenetration,
    ChechIntegrity_ContactArea,
    CheckIntegrity_CoulombConsistency
              ]

    for test in totest:
        res =  test()
        if  res.lower() != "ok" :
            return res

    return "OK"

if __name__ == '__main__':
    print(CheckIntegrity())