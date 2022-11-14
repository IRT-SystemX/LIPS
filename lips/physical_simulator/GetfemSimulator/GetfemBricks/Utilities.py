
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

import getfem as gf
gf.util_trace_level(1)

from lips.physical_simulator.GetfemSimulator.GetfemBricks.ModelTools import GetModelVariableValue,DefineModel,AssembleProblem,ExportTangentMatrix
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

modelVarByPhyField={
    PFN.displacement:"u",
    PFN.contactMultiplier:"lambda",
    PFN.contactUnilateralMultiplier:"lambda"
}

#Error evaluation
def ComputeL2Norm(mfu,field,mim):
	return gf.compute_L2_norm(mfu,field,mim)

def ComputeH1Norm(mfu,field,mim):
	return gf.compute_H1_norm(mfu,field,mim)

def Compute2DDeformedWheelCenter(holeZone,model,mfu,basicDofNodes):
    refIndices=mfu.basic_dof_on_region(holeZone)
    positions=np.transpose(basicDofNodes)[refIndices][1::2]
    pos_c=np.mean(positions,axis=0)

    field=GetModelVariableValue(model,modelVarByPhyField[PFN.displacement])
    field_x=field[refIndices][0::2]
    field_y=field[refIndices][1::2]

    disp_c=np.array([np.mean(field_x),np.mean(field_y)])
    return pos_c,disp_c

def ComputeMassMatrix(phyproblem):
    modelDummy=DefineModel()
    modelDummy.add_fem_variable(modelVarByPhyField[PFN.displacement], phyproblem.feSpaces["disp"])
    mim=phyproblem.integrMethods["standard"]
    modelDummy.add_linear_generic_assembly_brick(mim, 'u.Test_u')
    AssembleProblem(modelDummy)
    smatMass=ExportTangentMatrix(modelDummy)
    return smatMass

def ComputeMassMatrixOnBound(phyproblem,boundTag):
    mfu=phyproblem.feSpaces[PFN.displacement]
    indm = mfu.basic_dof_on_region(boundTag)
    mim=phyproblem.integrMethods["standard"]
    expr = 'M(#1,#2)+=comp(vBase(#1).vBase(#2))(:,i,:,i)'
    M = gf.asm_boundary(boundTag, expr, mim, mfu, mfu)
    M = gf.Spmat('copy', M, indm, list(range(M.size()[1])))
    return M, M.full()[:, indm]

def ComputeMultiplierMassMatrixOnBound(phyproblem,boundTag):
    boundRegion=phyproblem.refNumByRegion[boundTag]
    mflambda=phyproblem.feSpaces[PFN.contactUnilateralMultiplier]
    modelDummy=DefineModel()
    modelDummy.add_filtered_fem_variable("lambda", mflambda,boundRegion)
    mim_c=phyproblem.integrMethods["composite"]
    modelDummy.add_linear_generic_assembly_brick(mim_c, 'lambda*Test_lambda', boundRegion)
    AssembleProblem(modelDummy)
    matContact=ExportTangentMatrix(modelDummy)
    return sparse.csc_matrix(matContact)

def ComputeIntegralOverBoundary(model,expression,mim,region=-1):
    return gf.asm_generic(mim, 0, expression, region, model)

def GetRegionNodesIndices(primalSpace,dualSpace,region):
    refIndicesDual=dualSpace.basic_dof_on_region(region)
    coordDual=dualSpace.basic_dof_nodes().transpose()[refIndicesDual]

    refIndicesPrimal=primalSpace.basic_dof_on_region(region)
    coordPrimal=primalSpace.basic_dof_nodes().transpose()[refIndicesPrimal]

    y=cdist(coordPrimal,coordDual)
    dofInPrimal=refIndicesPrimal[np.where(y==0)[0]]
    return dofInPrimal

def ComputeMatNodesToContact(primalSpace,dualSpace,region):
    dofInP1=GetRegionNodesIndices(primalSpace,dualSpace,region)
    nlambda=len(dualSpace.basic_dof_on_region(region))
    nU=primalSpace.nbdof()
    matB=np.zeros((nlambda,nU))

    dofNum=np.arange(dofInP1.shape[0])
    dofYNum=dofNum[np.where(dofNum%2==1)]
    dofLambdaNum=np.arange(dofYNum.shape[0])
    matB[dofLambdaNum,dofInP1[dofYNum]]=1
    return matB

def Interpolate2DFieldOnSupport(model,originalSupport,originalField,targetSupport,enableExtrapolation=False):
   model.set_variable(modelVarByPhyField[PFN.displacement],originalField)
   args=[modelVarByPhyField[PFN.displacement],targetSupport,originalSupport,-1,enableExtrapolation]
   return model.interpolation(*tuple(args))

def InterpolateDisplacementOnSupport(model,originalSupport,cloudPoints):
   return model.interpolation(modelVarByPhyField[PFN.displacement], cloudPoints,originalSupport)

def ComputeDirichletNodalForces(phyproblem,dirichBoundary):
    mesh,mfu,mim=phyproblem.mesh,phyproblem.feSpaces[PFN.displacement],phyproblem.integrMethods["standard"]
    model=phyproblem.model
    nodalForces=EquivalentDirichletNodalForces(model=model,mesh=mesh,mim=mim,mfDisp=mfu,dirichBoundary=dirichBoundary)
    return nodalForces

def EquivalentDirichletNodalForces(model,mesh,mim,mfDisp,dirichBoundary):
    nodalForces=model.variable('mult_on_u')
    dummyModel=gf.Model('real')
    mflambda=gf.MeshFem(mesh, 2)
    mflambda.set_fem(gf.Fem('FEM_PK(2,1)'))

    dummyModel.add_filtered_fem_variable("ulambda", mflambda,dirichBoundary)
    dummyModel.add_linear_generic_assembly_brick(mim, 'ulambda.Test_ulambda', dirichBoundary)
    dummyModel.assembly()
    massMat=ExportTangentMatrix(dummyModel)

    newNodalForces=-massMat.dot(nodalForces)

    refIndices=mflambda.basic_dof_on_region(dirichBoundary)
    extendedNodalForces=np.zeros(mflambda.nbdof())
    extendedNodalForces[refIndices]=newNodalForces
    extendedNodalForces= gf.compute_interpolate_on(mflambda, extendedNodalForces, mfDisp)

    return extendedNodalForces

def ProjectSolOnMesh(coarseSol,solDegree,coarseMesh,refMesh):
    mfuCoarse=gf.MeshFem(coarseMesh,2)
    mfuCoarse.set_fem(gf.Fem('FEM_PK(2,%d)' % (solDegree,)));
    mfuRef=gf.MeshFem(refMesh,2)
    mfuRef.set_fem(gf.Fem('FEM_PK(2,%d)' % (solDegree,)));
    newSol= gf.compute_interpolate_on(mfuCoarse, coarseSol, mfuRef)
    return newSol