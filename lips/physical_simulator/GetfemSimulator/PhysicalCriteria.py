#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemMecaProblem
import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

class PhysicalCriteriaBase(metaclass=abc.ABCMeta):
    def __init__(self,problem):
        self.model=problem.model
        self.integrMethods=problem.integrMethods
        self.requiredFields=[]

    def AllRequiredFieldForCriteria(self,fieldNames):
        if not self.requiredFields:
            print("Warning: this criteria does not require any field to be computed")
            return True
        return set(self.requiredFields)<=set(fieldNames)

    def SetExternalSolutions(self,fields):
        if self.AllRequiredFieldForCriteria(fields.keys()):
            if set(fields.keys())>set(self.requiredFields):
                print("Warning: some fields provided are not required for the criteria "+str(self.__class__.__name__)+".\n Ignored Field:")
                print("\t",', '.join("{}".format(k) for k in list(set(fields.keys())-set(self.requiredFields))))
            for requiredField in self.requiredFields:
                modelVarName=PhySolver.modelVarByPhyField[requiredField]
                fieldValue=fields[requiredField]
                PhySolver.SetModelVariableValue(self.model,modelVarName,fieldValue)
        else:
            missingFields=list(set(self.requiredFields) - set(fields.keys()))
            missingFields=', '.join("{}".format(k) for k in missingFields)
            raise Exception("Criteria cannot be computed with this solution; fields missing:", missingFields)

    @abc.abstractmethod
    def ComputeValue(self):
        pass

class DeformedVolume(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(DeformedVolume,self).__init__(problem=problem)
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        return PhySolver.ComputeIntegralOverBoundary(model=self.model,expression="Det(Id(2)+Grad_u)",mim=self.integrMethods["standard"])


class UnilateralContactPressure(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(UnilateralContactPressure,self).__init__(problem=problem)
        self.refNumByRegion=problem.refNumByRegion
        self.requiredFields=[PFN.contactUnilateralMultiplier]

    def ComputeValue(self,contactBoundary):
        return PhySolver.ComputeIntegralOverBoundary(model=self.model,expression="lambda",mim=self.integrMethods["composite"],region=self.refNumByRegion[contactBoundary])

class FrictionalContactPressure(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(FrictionalContactPressure,self).__init__(problem=problem)
        self.refNumByRegion=problem.refNumByRegion
        self.requiredFields=[PFN.contactMultiplier]

    def ComputeValue(self,contactBoundary):
        return PhySolver.ComputeIntegralOverBoundary(model=self.model,expression="lambda",mim=self.integrMethods["composite"],region=self.refNumByRegion[contactBoundary])

class TotalElasticEnergy(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(TotalElasticEnergy,self).__init__(problem=problem)
        self.refNumByRegion=problem.refNumByRegion
        self.materials=problem.materials
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        elasticEnergy=0
        if len(self.materials)>1:
            raise Exception("Can only handle one material for now")

        for tagname,material in self.materials:
            if tagname=="ALL":
                materialLaw=material["law"]
                elasticEnergy+=PhySolver.ComputeTotalElasticEnergy(model=self.model,
                                                                behaviour_law=materialLaw,
                                                                mim=self.integrMethods["standard"])
                break
            else:
                materialLaw=material["law"]
                elasticEnergy+=PhySolver.ComputeTotalElasticEnergy(model=self.model,
                                                                behaviour_law=materialLaw,
                                                                mim=self.integrMethods["standard"],
                                                                region=self.refNumByRegion[tagname])
        return elasticEnergy

class MaxVonMises(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(MaxVonMises,self).__init__(problem=problem)
        self.materials=problem.materials
        self.mesh=problem.mesh
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        if len(self.materials)>1:
            raise Exception("Multi material not handled yet for this criteria!")

        vonMisesField=PhySolver.ComputeVonMises(model=self.model,
                                            material=self.materials[0],
                                            mim=self.integrMethods["standard"],
                                            mesh=self.mesh)
        return np.max(vonMisesField)

class MaxDisp(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(MaxDisp,self).__init__(problem=problem)
        self.mesh=problem.mesh
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        meshDimension=self.mesh.dim()
        displacements=PhySolver.GetModelVariableValue(self.model,"u")

        if displacements.shape[0] % meshDimension:
            raise Exception("Displacement size unconsistent with mesh dimension")

        fieldSizePerDof=displacements.shape[0]//meshDimension
        displacements_d=np.zeros((fieldSizePerDof,meshDimension))
        for dof in range(meshDimension):
            displacements_d[:,dof]=displacements[dof::meshDimension]
        return np.max(displacements_d,axis=0)

class PhysicalComplianceEquilibrium(PhysicalCriteriaBase):
    def __init__(self,problem):
        super(PhysicalComplianceEquilibrium,self).__init__(problem=problem)
        self.materials=problem.materials
        self.mesh=problem.mesh
        self.requiredFields=[PFN.displacement]

    def ComputeValue(self):
        if len(self.materials)>1:
            raise Exception("Multi material not handled yet for this criteria!")

        equilibriumResidual=PhySolver.ComputeEquilibriumResidual(model=self.model,
                                            material=self.materials[0],
                                            mim=self.integrMethods["standard"],
                                            mesh=self.mesh)
        return equilibriumResidual


import numpy.testing as npt
import numpy as np
def CheckIntegrity_NoFrictionContactProblem():
    BeamProblem=GetfemMecaProblem()
    BeamProblem.mesh,BeamProblem.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamProblem.materials=[["ALL", {"law":"SaintVenantKirchhoff","young":5e5,"poisson":0.3} ]]
    BeamProblem.dirichlet=[["LEFT",{"type" : "GlobalVector", "val_x":0.0, "val_y":0.0}] ]
    BeamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25}] ]
    BeamProblem.contact=[ ["BOTTOM",{"type" : "NoFriction","gap":0.0}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    multVariable=("lambda",PFN.contactUnilateralMultiplier,"BOTTOM") 
    BeamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    BeamProblem.BuildModel()
    BeamProblem.RunProblem()
    extSol={PFN.contactUnilateralMultiplier:BeamProblem.GetSolution(PFN.contactUnilateralMultiplier),
            PFN.displacement:BeamProblem.GetSolution(PFN.displacement)}

    unilatCriteria=UnilateralContactPressure(BeamProblem)
    oriUnilatPressure=unilatCriteria.ComputeValue(contactBoundary="BOTTOM")
    elasCriteria=TotalElasticEnergy(BeamProblem)
    oriElasticEnergy=elasCriteria.ComputeValue()

    BeamProblemCopy=type(BeamProblem)(BeamProblem)
    BeamProblemCopy.BuildModel()

    unilatCriteriaCopy=UnilateralContactPressure(BeamProblemCopy)
    unilatCriteriaCopy.SetExternalSolutions(extSol)
    newUnilatPressure=unilatCriteriaCopy.ComputeValue(contactBoundary="BOTTOM")
    npt.assert_almost_equal(oriUnilatPressure,newUnilatPressure)

    elasCriteriaCopy=TotalElasticEnergy(BeamProblem)
    elasCriteriaCopy.SetExternalSolutions(extSol)
    newElasticEnergy=elasCriteriaCopy.ComputeValue()
    npt.assert_almost_equal(oriElasticEnergy,newElasticEnergy)
    return "ok"

def CheckIntegrity_FrictionalContactProblem():
    BeamProblem2=GetfemMecaProblem()
    BeamProblem2.mesh,BeamProblem2.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamProblem2.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    BeamProblem2.dirichlet=[["LEFT",{"type" : "scalar", "Disp_X":0.0, "Disp_Y":0.0}] ]
    BeamProblem2.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-25.0}] ]
    BeamProblem2.contact=[ ["BOTTOM",{"type" : "Plane","gap":0.0,"fricCoeff":0.9}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    multVariable=("lambda",PFN.contactMultiplier,"BOTTOM") 
    BeamProblem2.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    BeamProblem2.BuildModel()
    BeamProblem2.RunProblem()
    extSol={PFN.contactMultiplier:BeamProblem2.GetSolution(PFN.contactMultiplier),
            PFN.displacement:BeamProblem2.GetSolution(PFN.displacement)}

    frictCriteria=FrictionalContactPressure(BeamProblem2)
    orifrictPressure=frictCriteria.ComputeValue(contactBoundary="BOTTOM")
    BeamProblemCopy=type(BeamProblem2)(BeamProblem2)
    BeamProblemCopy.BuildModel()

    frictCriteriaCopy=FrictionalContactPressure(BeamProblemCopy)
    frictCriteriaCopy.SetExternalSolutions(extSol)
    newfrictCriteria=frictCriteriaCopy.ComputeValue(contactBoundary="BOTTOM")
    npt.assert_almost_equal(orifrictPressure,newfrictCriteria)

    return "ok"

def CheckIntegrity_PureDisplacement():
    BeamProblem=GetfemMecaProblem()
    BeamProblem.mesh,BeamProblem.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamProblem.materials=[["ALL", {"law":"SaintVenantKirchhoff","young":5e5,"poisson":0.3} ]]
    BeamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_X":0.0, "Disp_Y":0.0}] ]
    BeamProblem.neumann=[["TOP",{"type" : "StandardNeumann", "fx":0.0, "fy":-2.}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    multVariable=("lambda",PFN.contactUnilateralMultiplier,"BOTTOM") 
    BeamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    BeamProblem.BuildModel()
    BeamProblem.RunProblem()
    extSol={PFN.displacement:BeamProblem.GetSolution(PFN.displacement)}

    criteriaToTest=[DeformedVolume,MaxVonMises,MaxDisp]
    criteriaVal=[None]*len(criteriaToTest)

    for criterionNum,criterionToTest in enumerate(criteriaToTest):
        mycriteria=criterionToTest(BeamProblem)
        criteriaVal[criterionNum]=mycriteria.ComputeValue()

    BeamProblemCopy=type(BeamProblem)(BeamProblem)
    BeamProblemCopy.BuildModel()

    for criterionNum,criterionToTest in enumerate(criteriaToTest):
        myCopyCriteria=criterionToTest(BeamProblemCopy)
        print(type(myCopyCriteria))
        myCopyCriteria.SetExternalSolutions(extSol)
        newValue=myCopyCriteria.ComputeValue()
        print(newValue)
        npt.assert_almost_equal(criteriaVal[criterionNum],newValue)

    return "ok"


def CheckIntegrity_PhysicalCompliance():
    BeamProblem=GetfemMecaProblem()
    BeamProblem.mesh,BeamProblem.refNumByRegion=PhySolver.GenerateSimpleMesh(meshSize=20.0)
    BeamProblem.materials=[["ALL", {"law":"LinearElasticity","young":5e5,"poisson":0.3} ]]
    BeamProblem.dirichlet=[["LEFT",{"type" : "scalar", "Disp_X":0.0, "Disp_Y":0.0}] ]
    BeamProblem.sources = [["ALL",{"type" : "Uniform","source_x":0,"source_y":-2e3}] ]
    femVariables={PFN.displacement: {"degree": 1, "dof": 2}}
    multVariable=("lambda",PFN.contactUnilateralMultiplier,"BOTTOM") 
    BeamProblem.Preprocessing(variables=femVariables,multiplierVariable=multVariable)
    BeamProblem.BuildModel()
    BeamProblem.RunProblem()
    #print(BeamProblem.GetSolution(PFN.displacement))


    mycriteria=PhysicalComplianceEquilibrium(BeamProblem)
    criteriaVal=mycriteria.ComputeValue()
    #print(criteriaVal)

    return "ok"

def CheckIntegrity():

    totest = [
    CheckIntegrity_NoFrictionContactProblem,
    CheckIntegrity_FrictionalContactProblem,
    CheckIntegrity_PureDisplacement,
    #CheckIntegrity_PhysicalCompliance
              ]

    for test in totest:
        res =  test()
        if  res.lower() != "ok" :
            return res

    return "OK"

if __name__ == '__main__':
    print(CheckIntegrity())