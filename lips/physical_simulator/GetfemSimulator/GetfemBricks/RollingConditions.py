#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from BasicTools.Helpers.Factory import Factory

import getfem as gf
gf.util_trace_level(1)

from lips.physical_simulator.GetfemSimulator.GetfemBricks.BrickBase import BrickBase
from lips.physical_simulator.GetfemSimulator.GetfemBricks.Utilities import Compute2DDeformedWheelCenter
from lips.physical_simulator.GetfemSimulator.GetfemBricks.ModelTools import GetModelVariableValue
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
from lips.physical_simulator.GetfemSimulator.Utilitaries import BasicDofNodesCoordinates

modelVarByPhyField={
    PFN.displacement:"u",
    PFN.contactMultiplier:"lambda",
    PFN.contactUnilateralMultiplier:"lambda"
}

class RollingFactory(Factory):
    _Catalog = {}
    _SetCatalog = set()
    def __init__(self):
        super(RollingFactory,self).__init__()

def CreateRollingBrick(name,ops=None):
    return RollingFactory.Create(name,ops)

class AddRollingCondition(BrickBase):
    def __init__(self):
        super(AddRollingCondition,self).__init__()
        self._name="Angle-piloted static displacement rolling on the rim"

    def Build(self,problemParams:dict,brickParams:dict):
        theta_R=brickParams["theta_Rolling"]
        model,mfu,mim = problemParams["model"],problemParams['feSpace'],problemParams["integrationMethod"]
        holeTagname,_=brickParams["regionTag"]
        holeZone=problemParams["tagMap"][holeTagname]
        g_rollingRHS=ComputeRollingCondition(holeZone,model,mfu,theta_R)

        model.add_initialized_fem_data('rollingRHS', mfu, g_rollingRHS)
        #Enforce new dirichlet condition using the field restricted on the rim
        idBrick=model.add_Dirichlet_condition_with_multipliers(mim, 'u', 1, holeZone, 'rollingRHS')
        return idBrick
RollingFactory.RegisterClass("AnglePiloted",AddRollingCondition)


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
    disp_m=GetModelVariableValue(model,modelVarByPhyField[PFN.displacement])
    field_x=disp_m[0::2]
    field_y=disp_m[1::2]

    g_XM=np.array((field_x,field_y))
    g_CM=g_CX+g_XM

    rotMat=np.array(([math.cos(theta_R),-math.sin(theta_R)],[math.sin(theta_R),math.cos(theta_R)]))
    #Evaluation of rotation value on whole mesh
    g_rotation=np.einsum('ij,jk',rotMat,g_CM)

    g_rollingRHS=g_translation + g_rotation - g_CM
    return g_rollingRHS


class AddDisplacementImposedRollingCondition(BrickBase):
    def __init__(self):
        super(AddDisplacementImposedRollingCondition,self).__init__()
        self._name="Angle-piloted & vertical quasi-static displacement rolling on the rim"

    def Build(self,problemParams:dict,brickParams:dict):
        dispRollingConditionParams={key:brickParams[key] for key in ["theta_Rolling","d","currentTime"]}
        problemRollingConditionParams={key:problemParams[key] for key in ["model","feSpace","dimension"]}
        holeTagname,_=brickParams["regionTag"]
        holeZone=problemParams["tagMap"][holeTagname]
        g_rollingRHS = ComputeDISRollingCondition(holeZone, problemRollingConditionParams, dispRollingConditionParams)

        model,mfu = problemParams["model"],problemParams['feSpace']
        model.add_initialized_fem_data('rollingRHS', mfu, g_rollingRHS)
        # Enforce new dirichlet condition using the field restricted on the rim
        mim=problemParams["integrationMethod"]
        idBrick = model.add_Dirichlet_condition_with_multipliers(mim, 'u', 1, holeZone, 'rollingRHS')
        return idBrick
RollingFactory.RegisterClass("DIS_Rolling",AddDisplacementImposedRollingCondition)


def ComputeDISRollingCondition(holeZone, problemRollingConditionParams, dispRollingConditionParams):
    theta_R,time=dispRollingConditionParams["theta_Rolling"],dispRollingConditionParams["currentTime"]
    dimension=problemRollingConditionParams["dimension"]

    rotMat=ComputeRotationMatrix(angle=theta_R*time,dimension=dimension)
    model, mfu=problemRollingConditionParams["model"],problemRollingConditionParams["feSpace"]
    basicDofNodes=mfu.basic_dof_nodes()
    x_c = Compute2DDeformedWheelCenter(holeZone, model, mfu, basicDofNodes)[0]
    d=dispRollingConditionParams["d"]
    speed = theta_R * (x_c[1] - d/3)

    dof_hole = mfu.basic_dof_on_region(holeZone)
    dof = mfu.basic_dof_nodes()
    pts = dof[:, dof_hole][:, 0::dimension]

    x_c[0] = 0.
    pts_rescaled = pts - np.repeat(x_c.reshape(-1, 1), pts.shape[1], axis=1)
    disp = np.dot(rotMat, pts_rescaled) - pts_rescaled
    disp[0, :] += speed * time
    disp[1, :] += -d

    rhs = np.zeros((dof.shape[1], ))
    for axeId in range(dimension):
        rhs[dof_hole[axeId::dimension]] = disp[axeId, :]
    return rhs

def ComputeRotationMatrix(angle,dimension):
    if dimension==2:
        rotMat = np.array(([math.cos(angle), math.sin(angle)],
                       [-math.sin(angle), math.cos(angle)]))
    elif dimension==3:
        rotMat = np.array(([math.cos(angle), math.sin(angle),0.0],
                       [-math.sin(angle), math.cos(angle),0.0],
                       [0.0,0.0,1.0]
                       ))
    else:
        raise Exception("Dimension not allowed")
    return rotMat

class AddForceImposedRollingCondition(BrickBase):
    def __init__(self):
        super(AddForceImposedRollingCondition,self).__init__()
        self._name="Angle-piloted & vertical quasi-static displacement rolling on the rim"

    def Build(self,problemParams:dict,brickParams:dict):
        model,mfu,mim = problemParams["model"],problemParams['feSpace'],problemParams["integrationMethod"]
        holeTagname,_=brickParams["regionTag"]
        holeZone=problemParams["tagMap"][holeTagname]
        model.add_initialized_data("Fx", brickParams["forc_x"])
        model.add_initialized_data("Fy", brickParams["forc_y"])
        model.add_initialized_data("time", brickParams['currentTime'])

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
RollingFactory.RegisterClass("FORC_Rolling",AddForceImposedRollingCondition)

class AddForceImposedRollingConditionPenalized(BrickBase):
    def __init__(self):
        super(AddForceImposedRollingConditionPenalized,self).__init__()
        self._name="Angle-piloted & vertical quasi-static displacement rolling on the rim"

    def Build(self,problemParams:dict,brickParams:dict):
        dispRollingConditionParams={key:brickParams[key] for key in ["theta_Rolling","d","currentTime"]}
        problemRollingConditionParams={key:problemParams[key] for key in ["model","feSpace","dimension"]}
        neumannTagname,_=brickParams["regionTag"]
        neumannZone=problemParams["tagMap"][neumannTagname]
        g_rollingRHS = ComputeDISRollingCondition(neumannZone, problemRollingConditionParams, dispRollingConditionParams)

        model, mfu=problemRollingConditionParams["model"],problemRollingConditionParams["feSpace"]
        model.add_initialized_fem_data('rollingRHS', mfu, g_rollingRHS)
        radius=brickParams["radius"]
        model.add_initialized_data('radius', [radius])
        penalization = brickParams['penalization']
        model.add_initialized_data('penalization', [penalization])

        integMethod = problemParams["integrationMethod"]
        idBrick = model.add_linear_term(integMethod, '-lambda_D.Test_u + radius * penalization * (rollingRHS - u).Test_lambda_D',
                                        neumannZone)
        return idBrick
RollingFactory.RegisterClass("FORC_Rolling_Penalized",AddForceImposedRollingConditionPenalized)
