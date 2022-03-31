#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This file introduce a module to describe a physical problem base using Getfem++ internal routines as the ingredients.
#As much as possible, we try to hide the implementation details involving specific Getfem syntax here so that 
#the user does not even have to know how the internal routines work to use them. Each and every functions/classes represent 
#a feature required in Getfem to run a computation

import numpy as np
import math
import random
import string
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import getfem as gf
gf.util('trace level', 1)
from lips.physical_simulator.GetfemSimulator.Utilitaries import BasicDofNodesCoordinates,WriteDataInCsv
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

modelVarByPhyField={
    PFN.displacement:"u",
    PFN.contactMultiplier:"lambda"
}

def Generate2DBeamMesh(meshSize,RefNumByRegion):
    NX = meshSize 
    mesh = gf.Mesh('regular simplices', np.arange(0,40+10/NX,10/NX), np.arange(0,10+10/NX,10/NX))

    #Boundaries definition
    P=mesh.pts()
    ctop=(abs(P[1,:]-10) < 1e-6)
    cbot=(abs(P[1,:]) < 1e-6)
    cright=(abs(P[0,:]-40) < 1e-6)
    cleft=(abs(P[0,:]) < 1e-6)

    pidtop=gf.compress(ctop, range(0, mesh.nbpts()))
    pidbot=gf.compress(cbot, range(0, mesh.nbpts()))
    pidright=gf.compress(cright, range(0, mesh.nbpts()))
    pidleft=gf.compress(cleft, range(0, mesh.nbpts()))

    ftop=mesh.faces_from_pid(pidtop)
    fbottom=mesh.faces_from_pid(pidbot)
    fright=mesh.faces_from_pid(pidright)
    fleft=mesh.faces_from_pid(pidleft)

    # Mark it as boundary
    mesh.set_region(RefNumByRegion["Left"], fleft)
    mesh.set_region(RefNumByRegion["Right"], fright)
    mesh.set_region(RefNumByRegion["Bottom"], fbottom)
    mesh.set_region(RefNumByRegion["Top"], ftop)
    return mesh

def get_random_string(length=15):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def GenerateSimpleMesh(meshSize):
    mesh=gf.Mesh('regular simplices', np.arange(0,4+1/meshSize,1/meshSize), np.arange(0,1+1/meshSize,1/meshSize))

    #Boundaries definition
    P=mesh.pts()
    ctop=(abs(P[1,:]-1) < 1e-6)
    cbot=(abs(P[1,:]) < 1e-6)
    cright=(abs(P[0,:]-4) < 1e-6)
    cleft=(abs(P[0,:]) < 1e-6)

    pidtop=gf.compress(ctop, range(0, mesh.nbpts()))
    pidbot=gf.compress(cbot, range(0, mesh.nbpts()))
    pidright=gf.compress(cright, range(0, mesh.nbpts()))
    pidleft=gf.compress(cleft, range(0, mesh.nbpts()))

    ftop=mesh.faces_from_pid(pidtop)
    fbot=mesh.faces_from_pid(pidbot)
    fright=mesh.faces_from_pid(pidright)
    fleft=mesh.faces_from_pid(pidleft)

    refNumByRegion={
               "TOP" : 1,
               "LEFT" : 2,
               "BOTTOM" : 3,
               "RIGHT" : 4
            }

    #Boundaries creation
    mesh.set_region(refNumByRegion["TOP"],ftop)
    mesh.set_region(refNumByRegion["LEFT"],fleft)
    mesh.set_region(refNumByRegion["BOTTOM"],fbot)
    mesh.set_region(refNumByRegion["RIGHT"],fright)
    return mesh,refNumByRegion


def GenerateWheelMesh(wheelDimensions,meshSize,RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    if r_inner >= r_ext:
        raise Exception("r_inner should be lower than r_ext")
    mo1 = gf.MesherObject('ball', [0., center], r_ext)
    mo2 = gf.MesherObject('ball', [0., center], r_inner)
    mo3 = gf.MesherObject('set minus', mo1, mo2)

    mesh = gf.Mesh('generate', mo3, meshSize, 2)
    mesh=AddWheelBoundaryConditions(mesh,wheelDimensions,RefNumByRegion)
    return mesh

def GetNumberOfNodes(mesh):
    return mesh.nbpts()

def GetNumberOfElements(mesh):
    return mesh.nbcvs()

def GetNodesInRegion(mesh,regionId):
    pts=mesh.pts().transpose()
    ptsIdInRegion=mesh.pid_in_regions(regionId)
    return pts[ptsIdInRegion]

def ExportMeshInVtk(mesh,meshFile):
    mesh.export_to_vtk(meshFile)

def AddWheelBoundaryConditions(mesh, wheelDimensions, RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    eps = 0.1
    # fb1 = mesh.outer_faces_in_box([-r_inner - eps, center - r_inner - eps], [r_inner + eps, center + r_inner + eps])  # Boundary of the hole
    # fb2 = mesh.outer_faces_with_direction([0., -1.], np.pi/2) # Contact boundary of the wheel
    fb2 = mesh.outer_faces_in_box([-center - eps, 0.-eps], [center+eps, center+eps]) # Contact boundary of the wheel
    # fb3 = mesh.outer_faces_in_box([-r_ext - eps, - eps], [r_ext + eps, center + r_ext + eps]) # Exterior boundary of tire

    fb1 = mesh.outer_faces_in_ball([0., center], r_inner + eps)
    fb3 = mesh.outer_faces_in_ball([0., center], r_ext + eps)
    fb4 = mesh.outer_faces_with_direction([0., 1.], np.pi/2)

    fb4 = mesh.outer_faces_with_direction([-np.cos(-np.pi/3), -np.sin(-np.pi/3)], 0.0)
    fb5 = mesh.outer_faces_with_direction([np.cos(np.pi/3), np.sin(np.pi/3)], 0.)

    mesh.set_region(RefNumByRegion["HOLE_BOUND"], fb1)
    mesh.set_region(RefNumByRegion["CONTACT_BOUND"], fb2)
    mesh.region_subtract(RefNumByRegion["CONTACT_BOUND"], RefNumByRegion["HOLE_BOUND"])
    mesh.set_region(RefNumByRegion["EXTERIOR_BOUND"], fb3)
    mesh.region_subtract(RefNumByRegion["EXTERIOR_BOUND"], RefNumByRegion["HOLE_BOUND"])

    return mesh

def GenerateWheelMeshRolling(wheelDimensions,meshSize,RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    if r_inner >= r_ext:
        raise Exception("r_inner should be lower than r_ext")
    mo1 = gf.MesherObject('ball', [0., center], r_ext)
    mo2 = gf.MesherObject('ball', [0., center], r_inner)
    mo3 = gf.MesherObject('set minus', mo1, mo2)

    mesh = gf.Mesh('generate', mo3, meshSize, 2)
    mesh=AddWheelBoundaryConditionsRolling(mesh,wheelDimensions,RefNumByRegion)
    return mesh

def AddWheelBoundaryConditionsRolling(mesh,wheelDimensions,RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    eps = 0.1
    # fb1 = mesh.outer_faces_in_box([-r_inner - eps, center - r_inner - eps], [r_inner + eps, center + r_inner + eps])  # Boundary of the hole
    fb2 = mesh.outer_faces_with_direction([0., -1.], np.pi/2) # Contact boundary of the wheel
    # fb3 = mesh.outer_faces_in_box([-r_ext - eps, - eps], [r_ext + eps, center + r_ext + eps]) # Exterior boundary of tire

    fb1 = mesh.outer_faces_in_ball([0., center], r_inner + eps)
    fb3 = mesh.outer_faces_in_ball([0., center], r_ext + eps)
    fb4 = mesh.outer_faces_with_direction([0., 1.], np.pi/2)

    mesh.set_region(RefNumByRegion["HOLE_BOUND"], fb1)
    mesh.set_region(RefNumByRegion["CONTACT_BOUND"], fb3)
    mesh.region_subtract(RefNumByRegion["CONTACT_BOUND"], RefNumByRegion["HOLE_BOUND"])
    mesh.set_region(RefNumByRegion["EXTERIOR_BOUND"], fb3)
    mesh.region_subtract(RefNumByRegion["EXTERIOR_BOUND"], RefNumByRegion["HOLE_BOUND"])
    if "NEUMANN_BOUND" in RefNumByRegion:
        mesh.set_region(RefNumByRegion["NEUMANN_BOUND"], fb4)

    return mesh


def ImportGMSHWheel(meshFile,wheelDimensions,RefNumByRegion):
    mesh=gf.Mesh('import', 'gmsh', meshFile)
    return AddWheelBoundaryConditions(mesh, wheelDimensions, RefNumByRegion)

def ImportGmshMesh(meshFile):
    return gf.Mesh('import', 'gmsh', meshFile)

#Definition Of FeSpaces
def DefineFESpaces(mesh, elements_degree, dof):
    mf = gf.MeshFem(mesh, dof)
    mf.set_classical_fem(elements_degree)
    return mf


#Integration Methods
def DefineIntegrationMethodsByOrder(mesh, order):
    return gf.MeshIm(mesh, order)


def DefineCompositeIntegrationMethodsByName(mesh,name):
    return gf.MeshIm(mesh, gf.Integ(name))

#Define model
def DefineModel():
    model=gf.Model('real')
    return model

def AddVariable(model,var,mfvar,boundary=None):
    if boundary is None:
        model.add_fem_variable(var, mfvar)
    else:
        model.add_filtered_fem_variable(var, mfvar,boundary)

def AddFixedSizeVariable(model,variableName,variableSize):
    model.add_variable(variableName, variableSize)

def AddConstraintWithMultipliers(model,variableName,multVariableName,matConstraint,rhsConstraint):
    idBrick=model.add_constraint_with_multipliers(variableName,multVariableName,matConstraint,rhsConstraint)
    return idBrick

#Source terms
def AddUniformSourceTerm(model,mim,params):
    sourceTermX,sourceTermY=params["source_x"],params["source_y"]
    model.add_initialized_data('f', [sourceTermX,sourceTermY])
    idBrick=model.add_source_term_brick(mim, 'u', 'f') 
    return idBrick


def AddVariableSourceTerm(model,mim,params):
    sourceTerm=params["source_term"].replace("X","X(1)").replace("Y","X(2)")
    idBrick=model.add_linear_generic_assembly_brick(mim, "-["+sourceTerm+"].Test_u")
    return idBrick

def ComputeMassMatrix(phyproblem):
    modelDummy=DefineModel()
    modelDummy.add_fem_variable("u", phyproblem.feSpaces["disp"])
    mim=phyproblem.IntegrMethod["standard"]
    modelDummy.add_linear_generic_assembly_brick(mim, 'u.Test_u')
    AssembleProblem(modelDummy)
    smatMass=ExportTangentMatrix(modelDummy)
    return smatMass

def ComputeMassMatrixOnBound(phyproblem,boundTag):
    mfu=phyproblem.feSpaces[PFN.displacement]
    indm = mfu.basic_dof_on_region(boundTag)
    mim=phyproblem.IntegrMethod["standard"]
    expr = 'M(#1,#2)+=comp(vBase(#1).vBase(#2))(:,i,:,i)'
    M = gf.asm_boundary(boundTag, expr, mim, mfu, mfu)
    M = gf.Spmat('copy', M, indm, list(range(M.size()[1])))
    return M, M.full()[:, indm]

def ComputeMultiplierMassMatrixOnBound(phyproblem,boundTag):
    boundRegion=phyproblem.refNumByRegion[boundTag]
    mflambda=phyproblem.feSpaces[PFN.contactUnilateralMultiplier]
    modelDummy=DefineModel()
    modelDummy.add_filtered_fem_variable("lambda", mflambda,boundRegion)
    mim_c=phyproblem.IntegrMethod["composite"]
    modelDummy.add_linear_generic_assembly_brick(mim_c, 'lambda*Test_lambda', boundRegion)
    AssembleProblem(modelDummy)
    matContact=ExportTangentMatrix(modelDummy)
    return sparse.csc_matrix(matContact)

#Behaviour laws
def AddLinearElasticity(model,mim,params):
    E,nu=params["young"],params["poisson"]
    clambda = E*nu/((1+nu)*(1-2*nu)) 
    cmu = E/(2*(1+nu))               
    clambdastar = 2*clambda*cmu/(clambda+2*cmu)
    model.add_initialized_data('cmu', [cmu])
    model.add_initialized_data('clambdastar', [clambdastar])
    idBrick=model.add_linear_generic_assembly_brick(mim, "clambdastar*(Div_u*Div_Test_u)+cmu*((Grad_u+(Grad_u)'):Grad_Test_u)")
    return idBrick

def AddIncompMooneyRivlin(model,mim,params):
    lawname = 'Incompressible Mooney Rivlin'
    c1,c2=params["MooneyRivlinC1"],params["MooneyRivlinC2"]
    model.add_initialized_data('paramsIMR', [c1,c2])
    idBrick=model.add_finite_strain_elasticity_brick(mim, lawname, 'u', 'paramsIMR')
    return idBrick

def AddSaintVenantKirchhoff(model,mim,params):
    lawname = 'SaintVenant Kirchhoff'
    clambda,cmu = params["clambda"],params["cmu"]
    model.add_initialized_data('paramsSVK', [clambda, cmu]);
    idBrick=model.add_finite_strain_elasticity_brick(mim, lawname, 'u', 'paramsSVK')
    return idBrick

#Contact Conditions
def AddUnilatContactWithFric(contactZone,model,mim,params):
    r=0.2 #Augmentation parameter for the augmented Lagrangian.  
    model.add_initialized_data('r', [r])
    model.add_initialized_data('N1', [0.,-1.0])
    model.add_initialized_data('gap', [params["gap"]])
    model.add_initialized_data('fric', [params["fricCoeff"]])
    idBrick1=model.add_linear_generic_assembly_brick(mim, '-lambda.(Test_u)', contactZone)
    idBrick2=model.add_linear_generic_assembly_brick(mim, '-(1/r)*lambda.Test_lambda', contactZone)
    idBrick3=model.add_nonlinear_generic_assembly_brick(mim, '(1/r)*Coulomb_friction_coupled_projection(lambda, N1, u,X(2)+gap-u.N1, fric, r).Test_lambda', contactZone)
    return idBrick1,idBrick2,idBrick3


def AddInclinedUnilatContactWithFric(contactZone,model,mim,params):
    found_angle=params["Found_angle"]
    shift=15.0
    r=0.2 #Augmentation parameter for the augmented Lagrangian.  
    model.add_initialized_data('r', [r])
    model.add_initialized_data('N1', [math.cos(found_angle),math.sin(found_angle)])
    model.add_initialized_data('shiftDist', [shift*(1+math.sin(found_angle))])
    model.add_initialized_data('fric', [params["fricCoeff"]])
    idBrick1=model.add_linear_generic_assembly_brick(mim, '-lambda.(Test_u)', contactZone)
    idBrick2=model.add_linear_generic_assembly_brick(mim, '-(1/r)*lambda.Test_lambda', contactZone)
    idBrick3=model.add_nonlinear_generic_assembly_brick(mim, '(1/r)*Coulomb_friction_coupled_projection(lambda, N1, u,-(X.N1-shiftDist)-u.N1, fric, r).Test_lambda', contactZone)
    return idBrick1,idBrick2,idBrick3

def AddUnilatContact(contactZone,model,mim,params):
    r=0.2 #Augmentation parameter for the augmented Lagrangian.  
    model.add_initialized_data('r', [r])
    model.add_initialized_data('N1', [0.,-1.0])
    model.add_initialized_data('gap', [params["gap"]])
    idBrick1=model.add_linear_generic_assembly_brick(mim, 'lambda*(Test_u.N1)', contactZone);
    idBrick2=model.add_nonlinear_generic_assembly_brick(mim, '-(1/r)*(-lambda + neg_part(-lambda-(r)*(u.N1-gap-X(2))))*Test_lambda', contactZone);
    return idBrick1,idBrick2


def AddPenalizedUnilatContactWithFric(contactZone,model,mfObstacle,mim,params):
    r=10
    model.add_initialized_data('r', [r])
    model.add_initialized_data('fric', [params["fricCoeff"]])
    OBS = mfObstacle.eval("y")
    model.add_initialized_fem_data('obstacle', mfObstacle, OBS)
    model.add_penalized_contact_with_rigid_obstacle_brick(mim, 'u', 'obstacle', 'r', 'fric', contactZone)

#EssentialBoundary
def AddDirichletCondition(dirichZone,dirichId,model,mim,params):
    amplitudeDisp,angleDisp=params["Disp_Amplitude"],params["Disp_Angle"]
    enforcedDisp=[amplitudeDisp*math.cos(angleDisp),amplitudeDisp*math.sin(angleDisp)]
    dirichVariable='DirichletData'+str(dirichId)
    model.add_initialized_data(dirichVariable, enforcedDisp)
    idBrick = model.add_Dirichlet_condition_with_multipliers(mim, 'u', 1, dirichZone, dirichVariable)
    return idBrick

def AddDirichletConditionWithSimplification(dirichZone,dirichId,model,mim,params):
    dirichVariable='DirichletData'+str(dirichId)
    model.add_initialized_data(dirichVariable, [params["val_x"],params["val_y"]])
    idBrick=model.add_Dirichlet_condition_with_simplification('u', dirichZone, dirichVariable)
    return idBrick

def AddDirichletConditionVector(dirichZone,dirichId,model,mim,params):
    enforcedDisp_x = params['uxy_vector'][:, 0]
    dataunitv_x = np.vstack((np.ones_like(enforcedDisp_x), np.zeros_like(enforcedDisp_x)))
    enforcedDisp_y = params['uxy_vector'][:, 1]
    dataunitv_y = np.vstack((np.zeros_like(enforcedDisp_y), np.ones_like(enforcedDisp_y)))
    ptsDisp = params['ptsxy_vector'].T
    model.add_initialized_data('DData_vec_x'+str(dirichId), enforcedDisp_x)
    model.add_initialized_data('dataunitv_x'+str(dirichId), dataunitv_x)
    model.add_initialized_data('DData_vec_y'+str(dirichId), enforcedDisp_y)
    model.add_initialized_data('dataunitv_y'+str(dirichId), dataunitv_y)
    model.add_initialized_data('pts_vec'+str(dirichId), ptsDisp)
    idBrick_ux = model.add_pointwise_constraints_with_multipliers('u', 'pts_vec'+str(dirichId), 'dataunitv_x'+str(dirichId), 'DData_vec_x'+str(dirichId))
    idBrick_uy = model.add_pointwise_constraints_with_multipliers('u', 'pts_vec'+str(dirichId), 'dataunitv_y'+str(dirichId), 'DData_vec_y'+str(dirichId))
    return idBrick_ux, idBrick_uy

def ComputeDirichletConditionRHS(dirichZone, params):
    pts_rom, u_rom, mfu = params["ptsxy"], params["uxy"], params['mfu']

    dof = mfu.basic_dof_nodes().T[0::2, :]
    dof_hole = mfu.basic_dof_on_region(dirichZone)
    myTree = cKDTree(pts_rom)
    _, inverse_map = myTree.query(dof, k=1)

    u_rom = u_rom[inverse_map, :]
    u_rom_final = np.zeros((2 * dof.shape[0], ))
    u_rom_final[0::2] = u_rom[:,0]
    u_rom_final[1::2] = u_rom[:, 1]
    g_rollingRHS = np.zeros((2 * dof.shape[0],))
    g_rollingRHS[dof_hole[0::2]] = u_rom_final[dof_hole[0::2]]
    g_rollingRHS[dof_hole[1::2]] = u_rom_final[dof_hole[1::2]]
    return g_rollingRHS

def AddDirichletConditionRHS(dirichZone, dirichId, model, mim, params):
    g_rollingRHS=ComputeDirichletConditionRHS(dirichZone, params)

    dirichVariable='DirichletData'+str(dirichId)
    mfu = params['mfu']
    model.add_initialized_fem_data(dirichVariable, mfu, g_rollingRHS)
    idBrick = model.add_Dirichlet_condition_with_multipliers(mim, 'u', 1, dirichZone, dirichVariable)
    return idBrick

def PrintModel(model):
    model.variable_list()

def PrintBricks(model):
    model.brick_list()

def AddRimRigidityNeumannCondition(neumannZone,neumannId,model,mfl,mim,params):
    pressure=[params["Force"]/(8*2*np.pi)]
    model.add_filtered_fem_variable('lambda_D',mfl, neumannZone)
    neumannVariable='F'+str(neumannId)
    model.add_initialized_data(neumannVariable, pressure)
    model.add_variable('alpha_D', 1)
    idBrick=model.add_linear_term(mim,\
        '-lambda_D.Test_u + (alpha_D*[0;1] - u).Test_lambda_D + (lambda_D.[0;1] + '+neumannVariable+')*Test_alpha_D + 1E-6*alpha_D*Test_alpha_D', neumannZone)
    return idBrick


def AddNeumannCondition(neumannZone,neumannId,model,mfl,mim,params):
    model.add_initialized_data('NeumannData'+str(neumannId), [params["fx"],params["fy"]])
    idBrick=model.add_source_term_brick(mim, 'u', 'NeumannData'+str(neumannId),neumannZone)
    return idBrick

def AddRollingCondition(holeZone,model,mfu,mim,params):
    theta_R=params["theta_Rolling"]
    g_rollingRHS=ComputeRollingCondition(holeZone,model,mfu,theta_R)

    model.add_initialized_fem_data('rollingRHS', mfu, g_rollingRHS)
    #Enforce new dirichlet condition using the field restricted on the rim
    idBrick=model.add_Dirichlet_condition_with_multipliers(mim, 'u', 1, holeZone, 'rollingRHS')
    return idBrick

def ComputeRollingCondition(holeZone,model,mfu,theta_R):
    #Compute center position and displacement
    allPositions=mfu.basic_dof_nodes()
    x_c,u_c=Compute2DDeformedWheelCenter(holeZone,model,mfu,allPositions)

    #Evaluation of translation value on whole mesh
    positions=np.transpose(allPositions)[1::2]
    coord_x,coord_y,_=BasicDofNodesCoordinates(positions)
    g_translation=np.repeat(u_c,coord_x.shape[0]).reshape((2,coord_x.shape[0]))

    #Evaluation of CM on whole mesh
    g_CXPos=np.array((coord_x,coord_y))
    g_newCenter=np.repeat(x_c+u_c,coord_x.shape[0]).reshape((2,coord_x.shape[0]))
    g_CX=g_CXPos-g_newCenter

    #Retrieve solution
    disp_m=GetModelVariableValue(model,"u")
    field_x=disp_m[0::2]
    field_y=disp_m[1::2]

    g_XM=np.array((field_x,field_y))
    g_CM=g_CX+g_XM

    rotMat=np.array(([math.cos(theta_R),-math.sin(theta_R)],[math.sin(theta_R),math.cos(theta_R)]))
    #Evaluation of rotation value on whole mesh
    g_rotation=np.einsum('ij,jk',rotMat,g_CM)

    g_rollingRHS=g_translation + g_rotation - g_CM
    return g_rollingRHS

def AddDisplacementImposedRollingCondition(holeZone,model,mfu,mim,params):
    theta_R = params["theta_Rolling"]
    d = params['d']
    current_time = params['currentTime']
    g_rollingRHS = ComputeDISRollingCondition(holeZone, model, mfu, theta_R, d, current_time)

    model.add_initialized_fem_data('rollingRHS', mfu, g_rollingRHS)
    # Enforce new dirichlet condition using the field restricted on the rim
    idBrick = model.add_Dirichlet_condition_with_multipliers(mim, 'u', 1, holeZone, 'rollingRHS')
    return idBrick

def ComputeDISRollingCondition(holeZone, model, mfu, theta_R, d, time):
    rotMat = np.array(([math.cos(theta_R * time), math.sin(theta_R * time)],
                       [-math.sin(theta_R * time), math.cos(theta_R * time)]))
    basicDofNodes=mfu.basic_dof_nodes()
    x_c = Compute2DDeformedWheelCenter(holeZone, model, mfu, basicDofNodes)[0]
    speed = theta_R * (x_c[1] - d/3)

    dof_hole = mfu.basic_dof_on_region(holeZone)
    dof = mfu.basic_dof_nodes()
    pts = dof[:, dof_hole][:, 0::2]

    x_c[0] = 0.
    pts_rescaled = pts - np.repeat(x_c.reshape(-1, 1), pts.shape[1], axis=1)
    disp = np.dot(rotMat, pts_rescaled) - pts_rescaled
    disp[0, :] += speed * time
    disp[1, :] += -d

    rhs = np.zeros((dof.shape[1], ))
    rhs[dof_hole[0::2]] = disp[0, :]
    rhs[dof_hole[1::2]] = disp[1, :]
    return rhs

def AddForceImposedRollingCondition(holeZone,model,mfu,mim,params):
    model.add_initialized_data("Fx", params["forc_x"])
    model.add_initialized_data("Fy", params["forc_y"])
    model.add_initialized_data("time", params['currentTime'])

    model.add_filtered_fem_variable('lambda_D',mfu, holeZone)
    model.add_variable('alpha_rot', 1)
    model.add_variable('alpha_y', 1)

    basicDofNodes=mfu.basic_dof_nodes()
    pos_c,_=Compute2DDeformedWheelCenter(holeZone,model,mfu,basicDofNodes)
    _,y_c=pos_c
    model.add_initialized_data("y_c", y_c)

    gwfl_rot='-lambda_D.Test_u + ([cos(alpha_rot*time),sin(alpha_rot*time);-sin(alpha_rot*time),cos(alpha_rot*time)]*(X-[0;y_c])-(X-[0;y_c])\
    + [alpha_rot * (y_c + alpha_y/3)*time;alpha_y] - u).Test_lambda_D +(lambda_D.[1;0] + Fx)*Test_alpha_rot + 1E-6*alpha_rot*Test_alpha_rot'
    gwfl_y='(lambda_D.[0;1] + Fy)*Test_alpha_y + 1E-6*alpha_y*Test_alpha_y'
    roll_gwfl=gwfl_y+gwfl_rot

    idBrick=model.add_nonlinear_term(mim,roll_gwfl,holeZone)
    return idBrick

def Compute2DDeformedWheelCenter(holeZone,model,mfu,basicDofNodes):
    refIndices=mfu.basic_dof_on_region(holeZone)
    positions=np.transpose(basicDofNodes)[refIndices][1::2]
    pos_c=np.mean(positions,axis=0)

    field=GetModelVariableValue(model,"u")
    field_x=field[refIndices][0::2]
    field_y=field[refIndices][1::2]

    disp_c=np.array([np.mean(field_x),np.mean(field_y)])
    return pos_c,disp_c


def DeleteBoundaryCondition(model,BCId):
    model.delete_brick(BCId)
    try:
        model.delete_variable("mult_on_u")
    except:
        model.delete_variable("mult_on_u_2")

def DeleteVariable(model,variableName):
    model.delete_variable(variableName)

def CleanModel(model):     
    model.clear()

def GetNbDof(model):
    return model.nbdof()

def AssembleProblem(model):
    model.assembly()

def Solve(model,max_iter,max_residual,noisiness=True):
    solveArgs=['max_res', max_residual, 'max_iter', max_iter]
    if noisiness:
        solveArgs+=['noisy']
    solverIt=model.solve(*tuple(solveArgs))
    return solverIt[0]<max_iter

def TwoStepsRollingSolve(problem):
    model,max_iter,max_residual=problem.model,problem.max_iter,problem.max_residual

    initVariablesName=['u','lambda_D']
    initVariablesVal=GetModelVariables(model,initVariablesName)

    model.disable_variable('alpha_rot')
    print("Solve first step")
    solverState=Solve(model,max_iter,max_residual)
    if not solverState:
        print("First step has failed")
        return solverState
    model.enable_variable('alpha_rot')

    SetModelVariables(model,initVariablesName,initVariablesVal)
    model.disable_variable('alpha_y')
    print("Solve second step")
    solverState=Solve(model,max_iter,max_residual)
    model.enable_variable('alpha_y')
    return solverState

def GetModelVariables(model,variableNames):
    sols=dict()
    for variableName in variableNames:
        sols[variableName]=GetModelVariableValue(model,variableName)
    return sols

def GetModelVariableValue(model,variableName):
    return model.variable(variableName)

def SetModelVariables(model,variableNames,variableValues):
    for variableName in variableNames:
        variableValue=variableValues[variableName]
        SetModelVariableValue(model,variableName,variableValue)

def SetModelVariableValue(model,variableName,variableValue):
    model.set_variable(variableName,variableValue)

def GetVariableDofInterval(model,variableModelName):
    varStart,varSize=model.interval_of_variable(variableModelName)
    return varStart,varStart+varSize

#Error evaluation
def ComputeL2Norm(mfu,field,mim):
	return gf.compute_L2_norm(mfu,field,mim)

def ComputeH1Norm(mfu,field,mim):
	return gf.compute_H1_norm(mfu,field,mim)

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

def ExportTangentMatrix(model):
    data=model.tangent_matrix().csc_val()
    rows,nbval=model.tangent_matrix().csc_ind()
    nbdof=rows.shape[0]-1
    return sparse.csc_matrix((data, nbval, rows), shape=(nbdof, nbdof))

def ExportRhs(model):
    return model.rhs()

def Interpolate2DFieldOnSupport(model,originalSupport,originalField,targetSupport):
   model.set_variable('u',originalField)
   return model.interpolation('u', targetSupport,originalSupport)

def InterpolateDisplacementOnSupport(model,originalSupport,cloudPoints):
   return model.interpolation('u', cloudPoints,originalSupport)
   
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

def AddExplicitRHS(model,varName,explicitRHS):
    idBrick=model.add_explicit_rhs(varName, explicitRHS)
    return idBrick

def ExtractFullAssembledSystem(model):
    sparseTangentMatrix,rightHandSide=ExportSparseTangentMatrix(model),ExtractRHS(model)
    return sparseTangentMatrix,rightHandSide

def ExtractRHS(model):
    return model.rhs()

def ExportSparseTangentMatrix(model):
    tangentMatrix=ExtractTangentMatrix(model)
    data=tangentMatrix.csc_val()
    rows,nbval=tangentMatrix.csc_ind()
    nbdof=rows.shape[0]-1
    return sparse.csc_matrix((data, nbval, rows), shape=(nbdof, nbdof))

def ExtractTangentMatrix(model):
    return model.tangent_matrix()

def ProjectSolOnMesh(coarseSol,solDegree,coarseMesh,refMesh):
    mfuCoarse=gf.MeshFem(coarseMesh,2)
    mfuCoarse.set_fem(gf.Fem('FEM_PK(2,%d)' % (solDegree,)));
    mfuRef=gf.MeshFem(refMesh,2)
    mfuRef.set_fem(gf.Fem('FEM_PK(2,%d)' % (solDegree,)));
    newSol= gf.compute_interpolate_on(mfuCoarse, coarseSol, mfuRef)
    return newSol

#Post processing
def ExportFieldInGmsh(filename,mfu,U,mfField,field,fieldName):
    mfu.export_to_pos(filename, mfu, U, 'Displacement',mfField,field,fieldName) 

def ExportFieldInVTK(filename,mfu,U,mfField,field,fieldName):
    mfu.export_to_vtk(filename, mfu, U, 'Displacement',mfField,field,fieldName) 

def ExportSingleFieldInGmsh(filename,mfField,field,fieldName):
    mfField.export_to_pos(filename, mfField,field,fieldName) 

def ExportSingleFieldInVTK(filename,mfField,field,fieldName):
    mfField.export_to_vtk(filename, mfField,field,fieldName) 

def ExportPrimalSolutionInGmsh(filename,feSpace,solutions):
    feSpace.export_to_pos(filename, feSpace, solutions, 'Displacement')

def ExportPrimalSolutionInVTK(filename,feSpace,solutions):
    feSpace.export_to_vtk(filename, feSpace, solutions, 'Displacement')

def ExportTransientPrimalSolutionInGmsh(filename,feSpace,solutions):
    fieldToExport=[[solution,'Displacement'+str(solId)] for solId,solution in enumerate(solutions)]
    fieldToExport=[item for sublist in fieldToExport for item in sublist]
    feSpace.export_to_pos(filename, feSpace, *fieldToExport)

def ExportTransientPrimalSolutionInVTK(filename,feSpace,solutions):
    fieldToExport=[[solution,'Displacement'+str(solId)] for solId,solution in enumerate(solutions)]
    fieldToExport=[item for sublist in fieldToExport for item in sublist]
    feSpace.export_to_vtk(filename, feSpace, *fieldToExport)

def ExportTransientPrimalSolutionInVTKSplitFiles(filename,feSpace,solutions):
    filenamePrefix=filename.split(".")[0]
    for solId,solution in enumerate(solutions):
        outputFile=filenamePrefix+str(solId)+".vtk"
        feSpace.export_to_vtk(outputFile, feSpace, solution,'Displacement')

def ExportTransientPrimalDualSolutionInGmsh(filename,feSpaces,solutions,contactBoundary):
    mfdisp,mfmult=feSpaces
    disps,mults=solutions
    refIndices=mfmult.basic_dof_on_region(contactBoundary)
    dispFieldToExport,multFieldToExport=[],[]
    for solId,(disp,mult) in enumerate(zip(disps,mults)):
        dispFieldToExport.append(disp)
        dispFieldToExport.append('Displacement'+str(solId))

        extendedLambda=np.zeros(mfmult.nbdof())
        extendedLambda[refIndices]=-mult
        multFieldToExport.append(mfmult)
        multFieldToExport.append(extendedLambda)
        multFieldToExport.append('Multiplier'+str(solId))

    mfdisp.export_to_pos(filename, mfdisp, *dispFieldToExport, *multFieldToExport)

def ExportTransientPrimalDualSolutionInVtk(filename,feSpaces,solutions,contactBoundary):
    filenamePrefix=filename.split(".")[0]
    mfdisp,mfmult=feSpaces
    disps,mults=solutions
    refIndices=mfmult.basic_dof_on_region(contactBoundary)
    for solId,(disp,mult) in enumerate(zip(disps,mults)):
        extendedLambda=np.zeros(mfmult.nbdof())
        extendedLambda[refIndices]=-mult
        
        outputFile=filenamePrefix+str(solId)+".vtk"
        mfdisp.export_to_vtk(outputFile, mfdisp, disp,'Displacement',mfmult,extendedLambda,'Multiplier')

def ExportSolutionInGmsh(filename,mflambda,lambdaC,contactZone,mfu,U,VM=None,mfVM=None):
    refIndices=mflambda.basic_dof_on_region(contactZone)
    extendedLambda=np.zeros(mflambda.nbdof())
    extendedLambda[refIndices]=-lambdaC

    if VM:
        mfu.export_to_pos(filename, mfVM, VM, 'Von Mises Stress', mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 
    else:
        mfu.export_to_pos(filename, mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 

def ExportSolutionInVTK(filename,mflambda,lambdaC,contactZone,mfu,U,VM=None,mfVM=None):
    refIndices=mflambda.basic_dof_on_region(contactZone)
    extendedLambda=np.zeros(mflambda.nbdof())
    extendedLambda[refIndices]=-lambdaC

    if VM:
        mfu.export_to_vtk(filename, mfVM, VM, 'Von Mises Stress', mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 
    else:
        mfu.export_to_vtk(filename, mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 

def GetBasicDof(mfu,regionTag=None,dofIds=None):
    if regionTag is not None:
        return mfu.basic_dof_on_region(regionTag)
    else:
        if dofIds is not None:
            return mfu.basic_dof_nodes(dofIds)
        return mfu.basic_dof_nodes()

def GetBasicCoordinates(mfu):
    allPositions=mfu.basic_dof_nodes()
    positions=np.transpose(allPositions)[1::2]
    return BasicDofNodesCoordinates(positions)

def GetUniqueBasicCoordinates(mfu):
    allPositions=mfu.basic_dof_nodes()
    coords=np.transpose(allPositions)[1::2].transpose()
    return coords

def GetSolutionAsField(mfu,displacements):
    data={}
    #Nodes original position
    data["Positions"]=GetBasicCoordinates(mfu)
    #Nodes displacements
    data["Ux"] = displacements[0::2]
    data["Uy"] = displacements[1::2]
    return data

def ExportSolutionInCSV(filename,outputformat,mfu,displacements):
    #Nodes original position
    allPositions=mfu.basic_dof_nodes()
    positions=np.transpose(allPositions)[1::2]
    data = {'Positions': positions}

    #Nodes displacements
    if outputformat == 'VectorSolution':
        data['Ux'] = displacements[0::2]
        data['Uy'] = displacements[1::2]
        dataNames=['Ux', 'Uy']
    else:
        data['Displacements'] = displacements
        dataNames='Displacements'
    WriteDataInCsv(outputformat, filename, data, dataNames)

def ExportTransientSolutionInCSV(filename,outputformat,mfu,solutions):
    filenamePrefix=filename.split(".")[0]
    nbSamples=len(solutions)
    maxIdDigits=len(str(nbSamples))+1
    for solId,solution in enumerate(solutions):
        fileId=str(solId).zfill(maxIdDigits)
        outputFile=filenamePrefix+fileId+".csv"
        ExportSolutionInCSV(outputFile,outputformat,mfu,solution)

def ExportNodalStressFieldInCSV(filename,outputformat,mfu,fieldVal):
    allPositions=mfu.basic_dof_nodes()
    positions=np.transpose(allPositions)[1::2]
    data = {'Positions': positions}

    #Nodes displacements
    if outputformat == 'VectorSolution':
        data['Fx'] = fieldVal[0::2]
        data['Fy'] = fieldVal[1::2]
        dataNames=['Fx', 'Fy']
    else:
        data['NodalStressField'] = fieldVal
        dataNames=['NodalStressField']
    WriteDataInCsv(outputformat,filename,data,dataNames)

def ExportMultipliersInCSV(filename,outputformat,mflambda,multipliers,contactZone):
    #Nodes original position
    refIndices=mflambda.basic_dof_on_region(contactZone)
    allPositions=mflambda.basic_dof_nodes()
    positions=np.transpose(allPositions)[refIndices]
    #Multipliers
    data={'Positions':positions,'Multipliers':multipliers}
    WriteDataInCsv(outputformat,filename,data,'Multipliers')